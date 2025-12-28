from train_scripts.trainer import *

import sys
import os
import torch.nn as nn
import gc
import torch
from typing import List, Optional, Union, Dict, Any

import argparse
import yaml
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import math
import time
import wandb
from tqdm import tqdm
from train_scripts.profiler import timer

from logger import print0 as print

if __name__ == '__main__':
    from train_scripts.train_functions import configure_optimizer_stage2, train_step

    parser = create_arg_parser()
    args = parser.parse_args()

    


    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    print(args)

    deepspeed.init_distributed()

    # 加载配置
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    # 设置设备和数据类型
    dtype = torch.bfloat16
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DeviceID = f'cuda:{args.local_rank}'
    args.DeviceID = DeviceID

   
    # 加载模型和分词器
    transformer_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'],
                                                             torch_dtype=dtype, device_map='cpu',low_cpu_mem_usage=True,trust_remote_code=True)
    

    args.freeze_attention = config['freeze_attention']
    args.hybrid_attention_layers = config['hybrid_attention_layers']
    args.freeze_hybrid_attention = config['freeze_hybrid_attention']
    args.allow_quant_frozen_layers = config['allow_quant_frozen_layers']
    args.quant_mode = config['quant_mode']
    args.peftmode = config['peftmode']
    args.peft_r = config['peft_r']
    args.peft_scaling = config['peft_scaling']
    args.peft_dropout = config['peft_dropout']
    args.mlp_quant_mode = config['mlp_quant_mode']
    args.bnb_optimizer_mode = config['bnb_optimizer_mode']
    args.transformer_layers = config['RWKV']['transformer_layers']
    args.disable_qk_norm = config['disable_qk_norm']

    args.architecture = config.get('architecture','hxa079')
    os.environ["architecture"] = args.architecture
    
    tokenizer = AutoTokenizer.from_pretrained(config['Llama']['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 设置参数
    args.my_pos_emb = 0
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_model.config.num_hidden_layers
    args.n_embd = transformer_model.config.hidden_size
    args.config = transformer_model.config
    
    args.dim_ffn = transformer_model.config.intermediate_size
    args.num_attention_heads = transformer_model.config.num_attention_heads
    args.num_key_value_heads = transformer_model.config.num_key_value_heads
    args.num_key_value_heads = transformer_model.config.num_key_value_heads
    args.rms_norm_eps = transformer_model.config.rms_norm_eps
    args.head_size_a = getattr(transformer_model.config, 'head_dim', transformer_model.config.hidden_size // transformer_model.config.num_attention_heads)
    args.dim_att = transformer_model.config.num_attention_heads * args.head_size_a
    args.is_attention_bias = getattr(transformer_model.config, 'attention_bias', True)
    args.is_attention_output_bias = getattr(transformer_model.config, 'attention_output_bias', False)
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_model.config.vocab_size
    args.layers = config['RWKV']['layers']
    args.pad_id = tokenizer.eos_token_id
    args.betas = (args.beta1, args.beta2)
    args.kl_weight = config['kl_weight']
    args.ce_weight = config['ce_weight']
    args.enable_AKL = config.get('enable_AKL', False)
    args.model_file = config['model_file']
    args.global_bsz = args.train_batch_size
    args.is_sft = config.get('is_sft', False)
    args.is_all_labels_kl = config.get('is_all_labels_kl', False)
    print(f'{transformer_model.config.num_hidden_layers}')

    if args.bnb_optimizer_mode > 0:
        args.deepspeed_stage = 1
        args.deepspeed_offload = False

    args.deepspeed_offload = False

    # 初始化混合模型
    teacher_attn_module_list = torch.nn.ModuleList()
    for layer_idx in range(transformer_model.config.num_hidden_layers):
        llama_layer = transformer_model.model.layers[layer_idx]
        teacher_attn_module_list.append(llama_layer.self_attn)
    for n,p in teacher_attn_module_list.named_parameters():
        p.requires_grad = False

    # FIXME - this is incorrect when att_dim is larger
    os.environ["RWKV_HEAD"] = str(int(args.n_embd // args.head_size_a))
    os.environ["RWKV_HEAD_SIZE_A"] = str(int(args.head_size_a))
    os.environ["RWKV_MICRO_BSZ"] = str(int(args.micro_bsz))
#     parser.add_argument('--quant_mode', type=str, default="int8", help='quant in peft mode except full')
#     parser.add_argument('--peftmode', type=str, default="full", help='peftmode full,lora,dora,bone')
#     parser.add_argument('--peft_r', type=int, default=32, help='peft block lora rank')
#     parser.add_argument('--peft_scaling', type=float, default=0.5, help='peft block lora scaling')
#     parser.add_argument('--peft_dropout', type=float, default=0.01, help='peft block lora dropout')

    os.environ['RWKV_ATTN_PEFTMODE'] = str(args.peftmode)
    os.environ['RWKV_ATTN_QUANT'] = str(args.quant_mode)
    os.environ['RWKV_ATTN_PEFT_R'] = str(args.peft_r)
    os.environ['RWKV_ATTN_PEFT_SCALING'] = str(args.peft_scaling)
    os.environ['RWKV_ATTN_PEFT_DROPOUT'] = str(args.peft_dropout)

    from model.hybrid_model import HybridModel

    model = HybridModel(transformer_model, args, tokenizer)

    force_cpu_memory_cleanup()
    
    # for name,param in model.named_parameters():
    #     #if 'mlp' in name:
    #         print(f'name = {name} {param.dtype} {param.device}')

    gc.collect()
    torch.cuda.empty_cache()

    def SearchTensor(model,keyname):
        for name,param in model.named_parameters():
            if keyname in name:
                return param
        return None
    
    weight_mul_r = 1.0
    weight_mul_k = 1.0
    weight_mul_v = 1.0
    weight_mul_o = 1.0 
    with torch.no_grad():
        for i in range(args.n_layer):
            if args.stage == 1:
                if i in args.transformer_layers:
                    weight_mul_r = 1.0
                    weight_mul_k = 1.0
                    weight_mul_v = 1.0
                    weight_mul_o = 1.0
                    
                else:
                    weight_mul_r = 0.5
                    weight_mul_k = 0.5
                    weight_mul_v = 0.3
                    weight_mul_o = 0.5 
            print(f'layer = {i} transfer to student')
            for name,param in model.named_parameters():
                #print(name)
                if f'model.layers.{i}.self_attn.student_attn' in name:
                    if 'receptance.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s*weight_mul_r)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'receptance.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_r)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    if 'key.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'key.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    
                    if 'value.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'value.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')


                    if 'output.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'output.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')




                    if 'q_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s*weight_mul_r)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'q_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_r)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    if 'k_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'k_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_k)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')

                    
                    if 'v_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'v_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.v_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_v)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')


                    if 'o_proj.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.weight')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    elif 'o_proj.bias' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.o_proj.bias')
                        if s != None:
                            if s.shape == param.shape:
                                param.copy_(s*weight_mul_o)
                                print('param copied from teacher')
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    









                    
                    if 'r_norm.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_norm.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    if 'q_norm.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.q_norm.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
                    if 'k_norm.weight' in name:
                        print(f'{name}')
                        s = SearchTensor(teacher_attn_module_list,f'{i}.k_norm.weight')
                        if s != None:
                            if s.shape == param.shape:
                                #param = s.clone()
                                param.copy_(s)
                                #print(param)
                                print('param copied from teacher')
                                #exit()
                            else:
                                print('shape is not same')
                        else:
                            print('not found')
    
    if args.local_rank == 0:
        print(model)
        # 打印几个关键参数的统计信息
        #print parameter:model.model.layers.27.self_attn.student_attn.ln_x.weight
        # for name, param in model.named_parameters():
        #     if name == pname:
        #         mean_of_param = param.mean().item()
        #         std_of_param = param.std().item()
        #         print(f"Parameter {name}: mean={mean_of_param:.6f}, std={std_of_param:.6f}")
    # 设置模型参数的训练状态
    print('all params are trainable')
    print(f'freeze mlp is {args.freeze_mlp}')

    # チェックポイントをロード
    checkpoint = torch.load(args.ckpt_file, map_location='cpu', mmap=True)

    # モデル形式に応じてstate_dictを取得
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 指定したモジュールを含まないパラメータのみをフィルタリング
    filtered_state_dict = {}

    # モデルの参照パラメータ
    model_state_dict = model.state_dict()

    filtered_state_dict = {}
    for key, value in state_dict.items():
        if any(excluded in key for excluded in ["embed", "lm_head", ".norm.", "mlp.weight", "qweight", "scales"]):
            print(f'{key} skipped [excluded]')
            continue

        if key not in model_state_dict:
            print(f'{key} skipped [missing in model]')
            continue

        if model_state_dict[key].shape != value.shape:
            print(f'{key} skipped [shape mismatch: ckpt{tuple(value.shape)} vs model{tuple(model_state_dict[key].shape)}]')
            continue

        # dtype 変換（浮動小数テンソルの場合のみ）
        if torch.is_floating_point(value):
            value = value.to(dtype=torch.bfloat16, copy=False)

        filtered_state_dict[key] = value
        print(f'{key} loaded (bfloat16)')

    # strict=False で欠けているキーがあってもロード続行
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    # オプション: ロード後の不一致を報告
    if missing:
        print(f'Missing keys in checkpoint: {missing}')
    if unexpected:
        print(f'Unexpected keys in checkpoint: {unexpected}')

    


    # 除外したパラメータ数と残ったパラメータ数を出力
    print(f"除外したパラメータ: {len(state_dict) - len(filtered_state_dict)}")
    print(f"ロードしたパラメータ: {len(filtered_state_dict)}")

    del checkpoint
    del filtered_state_dict
    del state_dict
    gc.collect()

    if args.grad_cp == 1:
        print('enable gradient checkpointing')
        model.model.gradient_checkpointing_enable()


    
    for name, param in model.named_parameters():
        Attention = 0
        for i in range(args.n_layer):
            t = f'layers.{i}.'
            if t in name and i in args.transformer_layers:
                Attention = 1
                break
            elif t in name:
                Attention = 0
                break
        if Attention == 0 and args.freeze_attention and ('receptance' in name or 'key' in name or 'value' in name):
            param.requires_grad = False
            print(f'{name} Frozen')
        elif Attention==1 and args.freeze_hybrid_attention and ('self_attn.student_attn' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name or 'q_norm' in name or 'k_norm' in name)):
            param.requires_grad = False
            print(f'{name} Frozen')
        elif args.freeze_mlp and 'mlp' in name and 'lora' not in name:
            param.requires_grad = False
            print(f'{name} Frozen')
        else:
            param.requires_grad = True
            print(f'{name} will Train')
            
    for name, param in model.named_parameters():
        if 'lm_head' in name or '.norm.' in name:
            param.requires_grad = False
            print(f'frozen {name}')

    lora_base_modules = set()
    if args.peftmode != 'full':
        print('freeze original weight if peft linears')
        
        # まず、LoRAモジュールを持つベースモジュール名を収集
        
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                # "blocks.0.att.receptance.lora_A.weight" -> "blocks.0.att.receptance"
                base_module_name = name.rsplit('.lora_', 1)[0]
                lora_base_modules.add(base_module_name)
                # LoRAパラメータ自体は学習可能にする
                param.requires_grad = True
                print(f'{name} LoRA param - will train!')

            elif 'bone' in name:
                # "blocks.0.att.receptance.lora_A.weight" -> "blocks.0.att.receptance"
                base_module_name = name.rsplit('.bone', 1)[0]
                lora_base_modules.add(base_module_name)
                # LoRAパラメータ自体は学習可能にする
                param.requires_grad = True
                print(f'{name} Bone param - will train!')
        
        # LoRAモジュールに対応する元のweight/biasをフリーズ
        for name, param in model.named_parameters():
            for base_module in lora_base_modules:
                # 元のweightをフリーズ
                if name == f"{base_module}.weight":
                    param.requires_grad = False
                    print(f'{name} Frozen (has LoRA)')
                # biasが存在する場合はフリーズ
                elif name == f"{base_module}.bias":
                    param.requires_grad = False
                    print(f'{name} Frozen (has LoRA)')

    # 最終的な学習可能パラメータの確認
    print("\n=== Final trainable parameters ===")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  {name}: {param.shape}")


    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nTrainable params: {trainable_params:,} / Total params: {total_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    print(f'current gpu memory BEFORE quant: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
    print(measure_model_memory(model))
    #Quant Phase
    if args.quant_mode != "none":
        for name, m in model.named_modules():
            Attention = 0
            for i in range(args.n_layer):
                t = f'layers.{i}.'
                if t in name and i in args.transformer_layers:
                    Attention = 1
                    break
                elif t in name:
                    Attention = 0
                    break
            #print(f'{name} {param.dtype}')
            if Attention == 0 and args.freeze_attention and ('self_attn.student_attn' in name and ('receptance' in name or 'key' in name or 'value' in name)):
                if hasattr(m, "quant") and callable(getattr(m, "quant")):
                    m.quant(args.quant_mode,DeviceID)
                    #print(f'{name} Quant on {DeviceID}. frozen RWKV')
            elif Attention == 1 and args.freeze_hybrid_attention and ('self_attn.student_attn' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name or 'q_norm' in name or 'k_norm' in name)):
                if hasattr(m, "quant") and callable(getattr(m, "quant")):
                    m.quant(args.quant_mode,DeviceID)
                    #print(f'{name} Quant on {DeviceID} frozen GQA')
            else:
                for base_module in lora_base_modules:
                    if name == f"{base_module}" and hasattr(m, "quant") and callable(getattr(m, "quant")):
                        m.quant(args.quant_mode,DeviceID)
                        #print(f'{name} Quant on {DeviceID} train peft')

    gc.collect()
    torch.cuda.empty_cache()


    train_dataloader, val_dataloader = get_dataloaders(args, tokenizer)

    # 设置DeepSpeed配置
    if args.deepspeed:
        if args.deepspeed_config:
            # 如果提供了 DeepSpeed 配置文件，直接加载它
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            # 否则，根据命令行参数创建配置
            ds_config = {
                "zero_force_ds_cpu_optimizer": True,
                "distributed_backend": "rccl",
                "train_batch_size": args.train_batch_size,
                "bf16": {
                    "enabled": True
                },
                #  "fp32_reduce_scatter": True,
                "zero_optimization": {
                    "stage": args.deepspeed_stage,

                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": False,
                        "buffer_count": 4,
                        'ratio':1.0
                    },

                #    "allgather_partitions": True,
                    #"sub_group_size": 1e7,
                    "overlap_comm": True,
                    "contiguous_gradients": False
                },
                "gradient_clipping": args.gradient_clip_val,
                "gradient_checkpointing": args.grad_cp == 1,
                "zero_allow_untested_optimizer": True,
                "gradient_accumulation_steps": args.accumulate_grad_batches if args.accumulate_grad_batches > 1 else None,
                # "wall_clock_breakdown": False,
                # "dump_state": True
            }
        if not args.deepspeed_offload:
            ds_config['zero_optimization']['offload_optimizer'] = None
            ds_config['zero_optimization']['offload_param'] = None
            ds_config['zero_force_ds_cpu_optimizer'] = False
            ds_config['zero_force_ds_cpu_initialization'] = False
        # 手动配置优化器
        print(f'configuring optimizer with args {args}')


        print("model to CUDA Device")
        model=model.to(device=DeviceID)
        print("done")

        print("optimizer settings")
        optimizer = configure_optimizer_stage2(model, args)
        print("done")
        if args.local_rank == 0:
            print(f'optimizer is {optimizer}')
            num_total_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            for n, p in model.named_parameters():
                print(f'{n} requires_grad = {p.requires_grad}')
            print(f'num_total_params: {num_total_params}, num_trainable_params: {num_trainable_params}, percent: {num_trainable_params / num_total_params * 100:.2f}%')
            #print current gpu memory
            print(f'current gpu memory BEFORE initializing deepspeed: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
            # model.model = torch.compile(model.model,fullgraph=True)
            # 初始化 DeepSpeed
            print(f'initializing deepspeed with config {ds_config}')
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,  
            optimizer=optimizer,
            config=ds_config
        )
        del model
        del transformer_model
        del optimizer
        del teacher_attn_module_list
        gc.collect()
        torch.cuda.empty_cache()
        
        # 添加验证代码
        # for name, param in model_engine.module.named_parameters():
        #     if name == pname:
        #         with deepspeed.zero.GatheredParameters(param):
        #             if args.local_rank == 0:  # 只在 rank 0 打印
        #                 print(f"Parameter {name}:")
        #                 print(f"  - mean: {param.mean().item():.6f} versus {mean_of_param:.6f}")
        #                 print(f"  - std: {param.std().item():.6f} versus {std_of_param:.6f}")
        #             break
            
            
        # if args.architecture == 'hxa07b':
        #     vfirst_holder = VFirstHolder(args.micro_bsz, args.max_seq_length,int(args.num_key_value_heads*2),args.head_size_a//2,device=DeviceID)
        #     vfirst_holder.requires_grad_(False)

        #     kfirst_holder = KFirstHolder(args.micro_bsz, args.max_seq_length,int(args.num_key_value_heads*2),args.head_size_a//2,device=DeviceID)
        #     kfirst_holder.requires_grad_(False)
        # else:
         
        # vfirst_holder = VFirstHolder(args.micro_bsz, args.max_seq_length,args.num_key_value_heads,args.head_size_a,device=DeviceID)
        # vfirst_holder.requires_grad_(False)

        # kfirst_holder = KFirstHolder(args.micro_bsz, args.max_seq_length,args.num_key_value_heads,args.head_size_a,device=DeviceID)
        # kfirst_holder.requires_grad_(False)
        
        # print(f'Zero 2 will hold the model in one GPU process,set the vfirst_holder to model_engine')
        # for layer_idx in args.layers:
        #     attn_wrapper = model_engine.module.model.model.layers[layer_idx].self_attn
        #     attn_wrapper.v_first_state = vfirst_holder
        #     attn_wrapper.k_first_state = kfirst_holder
        timer.initialize_with_engine(model_engine)
        #print current gpu memory
        if args.local_rank == 0:
            print(f'current gpu memory AFTER initializing deepspeed: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
        if args.stage == 2 and args.is_sft == False:
            if args.local_rank == 0:
                print(f'initializing teacher model')
                print(f'current gpu memory BEFORE initializing teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
     
            teacher_model_id = args.teacher_model_id
            if teacher_model_id is None:
                teacher_model_id = config['Llama']['model_id']
            print(f'initializing teacher model with id {teacher_model_id}')
            time.sleep(5)

            # # Int8量子化設定
            # from transformers import BitsAndBytesConfig
            # quantization_config = BitsAndBytesConfig(
            #     load_in_8bit=True,  # 4bitではなく8bitに変更
            #     int8_threshold=6.0,  # Int8量子化の閾値（デフォルト: 6.0）
            #     llm_int8_has_fp16_weight=False,  # FP16の重みを保持しない
            #     llm_int8_enable_fp32_cpu_offload=False  # CPU offloadを無効化
            # )

            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id,
                # quantization_config=quantization_config,  # 量子化設定を有効化
                torch_dtype=torch.bfloat16,  # Int8でも計算時の型指定は必要
                device_map=DeviceID,
                trust_remote_code=True
             #   low_cpu_mem_usage=True,
              #  attn_implementation="sdpa"
            )

            teacher_model.eval()

            #teacher_model = torch.compile(teacher_model)
            if args.local_rank == 0:
                print('freeze teacher_model')
                print(f'teacher_model is {teacher_model}')
            for name, param in teacher_model.named_parameters():
                param.requires_grad = False

            teacher_engine = teacher_model
      
            if args.local_rank == 0:
                print(f'current gpu memory AFTER initializing teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
                # 将处理好的teacher model设置到model_engine中
                # model_engine.module.set_teacher_model(teacher_engine.module)
                print(f'current gpu memory AFTER setting teacher model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
            # 清理不需要的引用
            #del teacher_model
            #teacher_engine=teacher_model
            gc.collect()
            torch.cuda.empty_cache()

        else:
            #Other stage we don't need teacher model
            #SFT or DPO
            teacher_engine = None
    else:
        # 如果不使用 DeepSpeed，使用普通的优化器
        print('not using deepspeed, EXIT')
        exit()

    # 只在主进程上初始化wandb
    if args.wandb and model_engine.global_rank == 0:
        print(f'init wandb, project is {args.wandb}, name is {args.run_name}')
        wandb.init(project=args.wandb, name=args.run_name, config=args)
        print(f'begin training with {args.max_epochs} epochs')
    # 初始化一些变量
    args.epoch_steps = len(train_dataloader) // (args.accumulate_grad_batches)
    global_step = 0
    last_log_time = time.time()
    token_per_step = args.max_seq_length * args.micro_bsz * args.world_size

    # 训练循环
    # 创建管理器实例
    terminate = False
    teacher_attn_manager = TeacherAttnManager(model_engine, args.layers)

    pbar = None
    trained_tokens = 0

    for epoch in range(args.max_epochs):
        if terminate:
            break

        model_engine.train()
        if model_engine.global_rank == 0:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(train_dataloader):
            
            lr, wd_now = on_train_batch_start(args, model_engine, global_step, epoch)

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # 前向传播
            loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step(model_engine, batch, args, teacher_engine, tokenizer)
            
            #CAUTION: The v_first will NEVER be synchronized for first batch. Just treat it as an outlier.

            model_engine.backward(loss)

            is_accumulation_step = (batch_idx + 1) % args.accumulate_grad_batches == 0
            grad_norm = None
            if is_accumulation_step:
                global_step += 1 
                # ★ DeepSpeed の global grad norm を取得
                try:
                    grad_norm = model_engine.get_global_grad_norm()
                except AttributeError:
                    grad_norm = None
               
            model_engine.step()

            # 每一步都调用 on_train_batch_end，但只在累积步骤结束时更新进度条
            last_log_time, pbar, trained_tokens = on_train_batch_end(
                args, batch_idx, model_engine,teacher_engine, loss.item(), teacher_loss, kl_loss, student_cross_entropy_loss,
                global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar, trained_tokens, grad_norm=grad_norm
            )

            if trained_tokens >= args.max_trained_tokens:
                terminate = True
                break
        



        # 保存检查点
        if args.output_dir:
            if args.deepspeed:
                
                # 在保存检查点的代码处使用上下文管理器
                with teacher_attn_manager.temporarily_remove_teacher_attn():
                    try:
                        print(f"Saving checkpoint to {args.output_dir} at epoch {epoch} rank {model_engine.global_rank}")
                        model_engine.save_checkpoint(args.output_dir, f"checkpoint-epoch{epoch}",exclude_frozen_parameters=True)
                    except Exception as e:
                        print(f"Error saving checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
