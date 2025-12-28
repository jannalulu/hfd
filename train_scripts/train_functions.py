import deepspeed
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from .profiler import time_function
import gc
import csv

from logger import print0 as print

def rank0_print(*args, **kwargs):
    if deepspeed.comm.get_rank() == 0:
        print(*args, **kwargs)
         

@time_function
def train_step(model, batch, args, teacher_engine=None, tokenizer=None, global_step=0, log_path="attn_log.csv"):
    # print(batch)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask'].to(torch.int32)      

    # 5. 非SFT模式的处理
    if args.stage == 2:
        if args.ce_weight > 0:
            if 'labels' in batch:
                labels = batch['labels']
                # 验证labels的维度
                if labels.shape != input_ids.shape:
                    raise ValueError(f"Labels shape {labels.shape} doesn't match input_ids shape {input_ids.shape}")
            else:
                # 直接创建左移的labels
                labels = torch.cat([input_ids[:, 1:], 
                                torch.full((input_ids.shape[0], 1), 
                                        tokenizer.pad_token_id, 
                                        device=input_ids.device)], dim=1)
        else:
            labels = None

        teacher_logits, teacher_loss = get_teacher_outputs(teacher_engine, input_ids, attention_mask, labels, args)
        gc.collect()
        torch.cuda.empty_cache()
        student_outputs = get_student_outputs(
            model, args, input_ids, labels, attention_mask)
        
        #compute_kl_loss_ultra_efficient
        loss, kl_loss, student_ce_loss = compute_kl_loss_ultra_efficient(
            student_outputs, teacher_logits, labels, args,attention_mask=attention_mask,temperature=1.0)
        # loss, kl_loss, student_ce_loss = compute_kl_loss(
        #     student_outputs, teacher_logits, labels, args,attention_mask=attention_mask)

        #loss = L2Wrap.apply(loss, student_outputs.logits)
 
    elif args.stage == 1:
        ##print('get student outputs')
        attention_hidden_states = []
        student_outputs = get_student_outputs(
            model, args, input_ids, None, attention_mask, attention_hidden_states=attention_hidden_states)
        #print('get_attn_loss')
        loss, kl_loss, student_ce_loss = get_attn_loss(model,
            args, attention_hidden_states=attention_hidden_states, step=global_step, log_path=log_path)
        teacher_loss = None
        
    return loss, teacher_loss, kl_loss, student_ce_loss
    
@time_function
# def get_attn_loss(model, args, student_outputs, step=None, log_path="attn_log.csv"):
#     """
#     HiddenState Alignment Loss (Scale-aware version)
#     - 方向合わせ（cosine / normalized MSE）
#     - ノルム合わせ（scale MSE or multiplicative scale）
#     """
#     dir_losses = []
#     norm_losses = []
#     layer_scales = []

#     for (s_h, t_h) in [student_outputs.attentions[i] for i in args.layers]:
#         # ----- Norm (length) -----
#         s_norm = s_h.norm(dim=-1, keepdim=True)
#         t_norm = t_h.norm(dim=-1, keepdim=True)

#         # スケール比（teacher / student）
#         scale_ratio = (t_norm.mean() / (s_norm.mean() + 1e-8)).item()
#         layer_scales.append(scale_ratio)

#         # ----- (1) 方向合わせ -----
#         # 正規化（方向）
#         s_dir = s_h / (s_norm + 1e-8)
#         t_dir = t_h / (t_norm + 1e-8)

#         # cosine 系の距離
#         dir_mse = F.mse_loss(s_dir, t_dir)
#         dir_losses.append(dir_mse)

#         # ----- (2) ノルム合わせ -----
#         # ノルムそのものを MSE
#         norm_mse = F.mse_loss(s_norm, t_norm)
#         norm_losses.append(norm_mse)

#     # ----- 最終層重視 -----
#     num_layers = len(dir_losses)
#     weights = torch.linspace(1.0, 1.5, num_layers).to(dir_losses[0].device)

#     # ----- 合成 Loss -----
#     # 方向合わせ（強め） + ノルム合わせ（弱め）
#     loss_dir = torch.sum(weights * torch.stack(dir_losses)) / torch.sum(weights)
#     loss_norm = torch.sum(weights * torch.stack(norm_losses)) / torch.sum(weights)

#     # ノルムは方向より弱めの係数（経験的に 0.05〜0.2 程度）
#     loss = loss_dir + 0.1 * loss_norm

#     # スケールの安定化のため係数
#     loss = loss# * 600.0

#     # ログ（Rank0）
#     kl_loss = None
#     student_cross_entropy_loss = None

#     if dist.is_initialized() and dist.get_rank() == 0 and step is not None:
#         mode = 'w' if step == 0 else 'a'
#         with open(log_path, mode=mode, newline='') as f:
#             writer = csv.writer(f)
#             if step == 0:
#                 header = (
#                     ["step"]
#                     + [f"layer_{i}_dirMSE" for i in args.layers]
#                     + [f"layer_{i}_normMSE" for i in args.layers]
#                     + [f"layer_{i}_scaleRatio" for i in args.layers]
#                 )
#                 writer.writerow(header)

#             row = (
#                 [step]
#                 + [float(l.item()) for l in dir_losses]
#                 + [float(l.item()) for l in norm_losses]
#                 + layer_scales
#             )
#             writer.writerow(row)

#     return loss, kl_loss, student_cross_entropy_loss

#Smerky Method + Mose Style
def get_attn_loss(model, args, attention_hidden_states, step=None, log_path="attn_log.csv"):
    raw_layer_losses, scaled_layer_losses, layer_scales = [], [], []

    #layers = range(len(attention_hidden_states))

    s_h = []
    t_h = []    
    for (s_h_i, t_h_i) in attention_hidden_states: #[attention_hidden_states[i] for i in args.layers]:
        # Norm Scale Ratio for record
        with torch.no_grad():
            s_norm = s_h_i.norm(dim=-1).mean()
            t_norm = t_h_i.norm(dim=-1).mean()
            scale_ratio = (t_norm / (s_norm + 1e-8)).item()
            layer_scales.append(scale_ratio)
    
            #Raw MSELoss for record
            raw_mse = F.mse_loss(s_h_i, t_h_i)
            raw_layer_losses.append(raw_mse)

        s_h.append( s_h_i )
        t_h.append( t_h_i )

    #Smerky's stage1 method
    # t_norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True) + 1e-12
    # s_normed = s / t_norm
    # t_normed = t / t_norm

    overall_scaling_constant = 175.0 / 64 # 175 was measured avg last layer norm for qwen3-8b, and 64 was sqrt(4096) hidden size for 8B
    overall_scaling_factor = overall_scaling_constant / torch.linalg.vector_norm(t_h[-1], dim=-1).mean()
    layer_scaling_factors = torch.linalg.vector_norm(t_h[-1], dim=-1).unsqueeze(0).mean(dim=[1,2]) / torch.linalg.vector_norm(torch.stack(t_h), dim=-1).mean(dim=[1,2])
    stacked_t_h = torch.stack(t_h) * layer_scaling_factors.view(-1,1,1,1)
    stacked_s_h = torch.stack(s_h) * layer_scaling_factors.view(-1,1,1,1)
    #training_loss = torch.linalg.vector_norm(stacked_t_h - stacked_s_h, dim=-1)
    training_loss = F.mse_loss(stacked_s_h,stacked_t_h)
    training_loss = training_loss * overall_scaling_factor

    loss = training_loss
    
    kl_loss = None
    student_cross_entropy_loss = None

    # ---- Rank0でCSVログ ----
    if dist.is_initialized() and dist.get_rank() == 0 and step is not None:
        mode = 'w' if step == 0 else 'a'
        with open(log_path, mode=mode, newline='') as f:
            writer = csv.writer(f)
            if step == 0:
                header = (
                    ["step"]
                    + [f"layer_{i}_rawMSE" for i in range(len(raw_layer_losses))]
                    + [f"layer_{i}_scaledMSE" for i in range(len(scaled_layer_losses))]
                    + [f"layer_{i}_scale" for i in range(len(layer_scales))]
                )
                writer.writerow(header)
            row = (
                [step]
                + [float(l.item()) for l in raw_layer_losses]
                + [float(l.item()) for l in scaled_layer_losses]
                + layer_scales
            )
            writer.writerow(row)

    return loss, kl_loss, student_cross_entropy_loss

def get_attn_loss_(model, args, attention_hidden_states, step=None, log_path="attn_log.csv"):
    """
    HiddenState Alignment Loss:
    - 各層ごとに MSE を計算
    - raw（未スケール）と scaled（正規化後）を両方ログ出力
    """
    raw_layer_losses, scaled_layer_losses, layer_scales = [], [], []

    for (s_h, t_h) in [attention_hidden_states[i] for i in args.layers]:
        # Norm Scale Ratio for record
        with torch.no_grad():
            s_norm = s_h.norm(dim=-1).mean()
            t_norm = t_h.norm(dim=-1).mean()
            scale_ratio = (t_norm / (s_norm + 1e-8)).item()
            layer_scales.append(scale_ratio)
    
            #Raw MSELoss for record
            raw_mse = F.mse_loss(s_h, t_h)
            raw_layer_losses.append(raw_mse)

        #Scaled Loss
        s_normed = F.normalize(s_h, dim=-1)
        t_normed = F.normalize(t_h, dim=-1)
        scaled_mse = F.mse_loss(s_normed, t_normed)
        scaled_layer_losses.append(scaled_mse)

    # Weight importance for last layer
    num_layers = len(raw_layer_losses)
    weights = torch.linspace(1.0, 1.5, num_layers).to(raw_layer_losses[0].device)
    # total loss
    ManualLossScaling = 600.0
    loss = torch.sum(weights * torch.stack(scaled_layer_losses)) / torch.sum(weights) * ManualLossScaling

    kl_loss = None
    student_cross_entropy_loss = None

    # ---- Rank0でCSVログ ----
    if dist.is_initialized() and dist.get_rank() == 0 and step is not None:
        mode = 'w' if step == 0 else 'a'
        with open(log_path, mode=mode, newline='') as f:
            writer = csv.writer(f)
            if step == 0:
                header = (
                    ["step"]
                    + [f"layer_{i}_rawMSE" for i in args.layers]
                    + [f"layer_{i}_scaledMSE" for i in args.layers]
                    + [f"layer_{i}_scale" for i in args.layers]
                )
                writer.writerow(header)
            row = (
                [step]
                + [float(l.item()) for l in raw_layer_losses]
                + [float(l.item()) for l in scaled_layer_losses]
                + layer_scales
            )
            writer.writerow(row)

    return loss, kl_loss, student_cross_entropy_loss

@time_function
def get_student_outputs(model, args, input_ids, labels, attention_mask, attention_hidden_states = None):
    student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            labels=labels, use_cache=False,
            attention_hidden_states=attention_hidden_states)#, 
            #output_attentions=args.stage==1)
        
    return student_outputs
@time_function
def get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args):
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=False, use_cache=False) #, use_cache=False, output_hidden_states=False
        teacher_logits = teacher_outputs.logits
        teacher_loss = teacher_outputs.loss
 
    return teacher_logits,  teacher_loss


@time_function #Modified Temperature
def compute_kl_loss(student_outputs, teacher_logits, labels, args, attention_mask=None, chunk_size=4096,temperature=1.0):
    student_logits = student_outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    vocab_student = student_logits.shape[-1]
    vocab_teacher = teacher_logits.shape[-1]

    # Truncate teacher logits if necessary
    if vocab_teacher > vocab_student:
        teacher_logits = teacher_logits[:, :, :vocab_student]
    
    # 温度パラメータの取得（argsにtemperature属性があると仮定）
    #temperature = getattr(args, "temperature", 2.0)

      
    # 温度スケーリングを適用したロジットの計算
    #student_logits_scaled = student_logits / temperature
    
    # Compute softmax for student and teacher with temperature scaling
    log_probs_student = F.log_softmax(student_logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    with torch.no_grad():
        teacher_logits_scaled = teacher_logits / temperature
        targets = F.softmax(teacher_logits_scaled, dim=-1)    # [batch_size, seq_len, vocab_size]
    
    # Compute KL divergence without reduction
    kl_div_all = F.kl_div(
        log_probs_student,
        targets,
        reduction='none'  # Keep the full tensor to apply mask
    )  # [batch_size, seq_len, vocab_size]
    
    # Sum across vocabulary dimension first
    kl_div_per_token = kl_div_all.sum(dim=-1)  # [batch_size, seq_len]
    
    if attention_mask is not None:
        # Apply attention mask and compute mean only over attended positions
        masked_kl = kl_div_per_token * attention_mask
        kl_loss = masked_kl.sum() / (attention_mask.sum() + 1e-6)  # Add small epsilon for numerical stability
    else:
        # If no mask provided, take mean over all tokens
        kl_loss = kl_div_per_token.mean()
    
    # 温度スケーリングによる勾配補正
    #kl_loss = (temperature ** 2) * kl_loss

    del log_probs_student, targets, kl_div_all, kl_div_per_token
    
    # Get cross entropy loss from student outputs
    student_cross_entropy_loss = student_outputs.loss
    
    # Combine losses using weights from args
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    
    del student_logits, teacher_logits, labels
    if attention_mask is not None:
        del attention_mask
    return loss, kl_loss, student_cross_entropy_loss
@time_function
def compute_kl_loss_ultra_efficient(student_outputs, teacher_logits, labels, args, attention_mask=None, chunk_size=1024, temperature=1.0):
    """
    勾配を正しく保持する極限最適化版
    """
    student_logits = student_outputs.logits
    vocab_student = student_logits.shape[-1]
    vocab_teacher = teacher_logits.shape[-1]
    
    if vocab_teacher > vocab_student:
        teacher_logits = teacher_logits[:, :, :vocab_student]
    
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # 勾配を保持するテンソルとして累積
    kl_sum = torch.zeros(1, device=student_logits.device, requires_grad=True)
    total_tokens = 0
    
    if args.kl_weight > 0:
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            
            # チャンク抽出
            student_chunk = student_logits[:, i:end_idx, :]
            teacher_chunk = teacher_logits[:, i:end_idx, :]
            
            # KL計算
            log_probs = F.log_softmax(student_chunk, dim=-1)
            
            with torch.no_grad():
                teacher_scaled = teacher_chunk / temperature
                targets = F.softmax(teacher_scaled, dim=-1)
            
            kl_div = F.kl_div(log_probs, targets, reduction='none')
            kl_per_token = kl_div.sum(dim=-1)  # [batch_size, chunk_size]
            
            # Attention mask適用
            if attention_mask is not None:
                mask_chunk = attention_mask[:, i:end_idx]
                masked_kl = kl_per_token * mask_chunk
                kl_sum = kl_sum + masked_kl.sum()
                total_tokens += mask_chunk.sum().item()
            else:
                kl_sum = kl_sum + kl_per_token.sum()
                total_tokens += kl_per_token.numel()
            
            # 中間テンソル削除
            del student_chunk, teacher_chunk, log_probs, targets, kl_div, kl_per_token
            if attention_mask is not None:
                del mask_chunk, masked_kl
            
            #torch.cuda.empty_cache()
    
    # 正規化（勾配保持）
    kl_loss = kl_sum / (total_tokens + 1e-6)
    
    # Cross entropy loss
    if args.ce_weight > 0:
        student_cross_entropy_loss = student_outputs.loss
    else:
        student_cross_entropy_loss = 0.0
    
    # Combine losses
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    
    # クリーンアップ
    del student_logits, teacher_logits, labels, kl_sum
    if attention_mask is not None:
        del attention_mask
    
    return loss, kl_loss, student_cross_entropy_loss

def configure_optimizer_stage2(model, args):
    lr_decay = set()
    lr_0x = set()
    lr_1x = set()
    lr_2x = set()
    #print(model.keys())
    for n, p in model.named_parameters():
        skiptensor = False
        #if 'embed' in n or 'lm_head' in n or '.norm.' in n:
        if 'embed' in n:
            skiptensor = True
        if skiptensor:
            print(f'name={n} grad={p.requires_grad}')
        if not p.requires_grad:
            print(f'{n} not train skipp optimizer')
            continue
        if ('attn.w0' in n ): # and (args.layerwise_lr > 0): #or 'attn.w1' in n or 'attn.w2' in n
            lr_2x.add(n)
            print(f'{n} 2x Learning Rate! Lets RocknLOL! ')
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
            if skiptensor:
                print(f'{n} zero lr')
                lr_0x.add(n)
            else:
                lr_decay.add(n)
                print(f'{n} lr decay')
        else:
            #print(f'{n}')
            if skiptensor:
                print(f'{n} zero lr')
                lr_0x.add(n)
            else:
                print(f'{n} 1x LR')
                lr_1x.add(n)

    #exit()
    lr_decay = sorted(list(lr_decay))
    lr_0x = sorted(list(lr_0x))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
            {"params": [param_dict[n] for n in lr_0x], "weight_decay": 0.0, "my_lr_scale": 0.0},
        ]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
    #print(optim_groups)

    optimizer = AdamW(optim_groups, lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    return optimizer


def configure_optimizer(model, args):
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
        #     lr_1x.add(n)
        # elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
        #     lr_2x.add(n)
        # elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
        #     lr_2x.add(n)
        # elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
        #     lr_1x.add(n)
        # elif ("time_first" in n) and (args.layerwise_lr > 0):
        #     lr_3x.add(n)
        # 
        if ('attn.w0' in n): # and (args.layerwise_lr > 0): # or 'attn.w1' in n or 'attn.w2' in 
            lr_2x.add(n)
            print(f'{n} 2x Learning Rate! Lets RocknLOL! ')
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
            lr_decay.add(n)
            print(f'{n} LR Decay')
        else:
            lr_1x.add(n)
            print(f'{n} 1x LR')

        
    #exit()

    lr_decay = sorted(list(lr_decay))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

    optimizer = AdamW(optim_groups, lr=args.lr_init, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    return optimizer
