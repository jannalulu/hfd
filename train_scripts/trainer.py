import sys
import os
import torch.nn as nn
import gc
import torch
from typing import List, Optional, Union, Dict, Any

from logger import print0 as print

def setup_env():
    # parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # model_path = os.path.join(parent_dir, 'model')
    # sys.path.append(model_path)
    # sys.path.append(parent_dir)
    # print(f'add path: {model_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = os.environ.get('RWKV_T_MAX', '4096')
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    
    os.environ['RWKV_CTXLEN'] = os.environ.get('RWKV_CTXLEN', '4096')
    if 'WKV' not in os.environ:
        os.environ['WKV'] = ''
    if "RWKV_TRAIN_TYPE" not in os.environ:
        os.environ["RWKV_TRAIN_TYPE"] = ''
    RWKV_VERSION = os.environ.get('RWKV_VERSION', 'v7')
    if RWKV_VERSION == 'v7':
        os.environ["RWKV_MY_TESTING"]='x070'
    else:
        os.environ["RWKV_MY_TESTING"]='x060'
    print(f'RWKV_VERSION is {RWKV_VERSION}')
    
setup_env()

import argparse

import math
import time
import wandb
from tqdm import tqdm
from .profiler import timer

import datasets

def measure_model_memory(model: torch.nn.Module, detailed: bool = True) -> Dict[str, float]:
    """
    モデルのVRAM使用量を測定
    
    Args:
        model: 測定対象のモデル
        detailed: 詳細な内訳を表示するか
    
    Returns:
        メモリ使用量の辞書 (GB単位)
    """
    # ガベージコレクションを実行してクリーンな状態にする
    gc.collect()
    torch.cuda.empty_cache()
    
    total_size = 0
    param_size = 0
    buffer_size = 0
    
    # パラメータのメモリ使用量
    for name, param in model.named_parameters():
        if param.is_cuda:
            size = param.numel() * param.element_size()
            param_size += size
            if detailed:
                size_mb = size / 1024**2
                print(f"Parameter {name}: {param.shape}, {param.dtype}, {size_mb:.2f} MB")
    
    # バッファのメモリ使用量
    for name, buffer in model.named_buffers():
        if buffer.is_cuda:
            size = buffer.numel() * buffer.element_size()
            buffer_size += size
            if detailed:
                size_mb = size / 1024**2
                print(f"Buffer {name}: {buffer.shape}, {buffer.dtype}, {size_mb:.2f} MB")
    
    total_size = param_size + buffer_size
    
    return {
        'total_gb': total_size / 1024**3,
        'param_gb': param_size / 1024**3,
        'buffer_gb': buffer_size / 1024**3,
        'total_mb': total_size / 1024**2,
        'param_mb': param_size / 1024**2,
        'buffer_mb': buffer_size / 1024**2,
    }

def exclude_int8_params_from_zero(model):
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:
            print(f"[ZeRO Exclude] Excluding int8 param from ZeRO: {name}")
            param._no_zero3 = True

def create_arg_parser():
    node_rank = int(os.environ.get('NODE_RANK', 0))
    num_gpus = int(os.environ.get('NUM_GPUS', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 7))
    print(f'node_rank: {node_rank}, num_gpus: {num_gpus}, world_size: {world_size}')
    parser = argparse.ArgumentParser(description='MLM trainer')
    parser.add_argument('--config_file', type=str,default='configs/test_hybrid.yaml', help='training config file')
    parser.add_argument('--preprocessed_data',type=str,nargs='+',help='preprocessed data directory')
    parser.add_argument('--raw_data',type=str,nargs='+',help='raw data directory')
    parser.add_argument('--need_to_pack',action='store_true',default=False,help='whether to pack the input with other sample to fill the sample to max length')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/tmp',help='directory to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length to train the model')
    parser.add_argument('--num_devices', type=int, default = 1,help='number of devices to train the model')
    parser.add_argument('--has_group_norm', action='store_true',default=False,help='whether the Time Mixer has group norm')
    parser.add_argument('--gate_free',action='store_true',default=False,help='whether the Time Mixer has gate free')
    parser.add_argument('--min_len', type=int, default=0, help='minimum length of the input')
    parser.add_argument('--max_len', type=int, default=4096, help='maximum length of the input')
    parser.add_argument('--freeze_mlp', action='store_true',default=False,help='freeze the mlp layer')
    parser.add_argument('--teacher_model_id', type=str, default=None, help='teacher model id used to distill in stage2')
    
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate in the model')
    parser.add_argument('--grad_cp', type=int, default=0, help='gradient checkpoint in the model')
    parser.add_argument('--save_per_batches', type=int, default=10000, help='number of batches to save the model')
    parser.add_argument('--my_exit', type=int, default=300, help='exit condition in the model')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay in the model')
    parser.add_argument('--lr_init', type=float, default=6e-4, help='initial learning rate in the model')
    parser.add_argument('--lr_final', type=float, default=1e-5, help='final learning rate in the model')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2 parameter in the Adam optimizer')
    parser.add_argument('--layerwise_lr', type=float, nargs='+', default=1, help='layerwise learning rate in the model')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon parameter in the Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=50, help='warmup steps in the model')
    parser.add_argument('--epoch_begin', type=int, default=0, help='beginning epoch for the training')
    parser.add_argument('--epoch_save', type=int, default=1, help='number of epochs after which the model is saved')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum number of epochs for the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='number of epochs after which the validation is checked')
    parser.add_argument('--val_check_interval', type=int, default=5000, help='number of epochs after which the validation is checked')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='number of validation steps for sanity check at the beginning of training')
    parser.add_argument('--log_every_n_steps', type=int, default=5000, help='number of steps after which the training progress will be logged')
    parser.add_argument('--enable_checkpointing', type=bool, default=False, help='flag to enable checkpointing')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate before performing a backward/update pass')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='maximum gradient norm')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--micro_bsz', type=int,default=2, help='micro batch size for training')
    parser.add_argument('--global_bsz', type=int, help='real batch size for training')
    parser.add_argument('--my_pile_stage', type=int, default=0, help='pile stage in the model')
    #parser.add_argument('--my_pile_edecay', type=float, default=0, help='pile exponential decay in the model')
    parser.add_argument('--weight_decay_final', type=float, default=-1, help='final weight decay in the model')
    parser.add_argument('--proj_dir', type=str, help='project directory to save the model and logs')
    parser.add_argument('--eval_every_steps', type=int, default=100, help='number of steps after which the model is evaluated')
    parser.add_argument('--wandb', type=str, default='hybrid_trainer', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='hybrid_trainer_a800', help='run name for wandb logging')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_2_offload', help='strategy for distributed training')
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument('--my_qa_mask', type=int, default=0)
    parser.add_argument('--optim',type=str,default='adam',help='optimizer')
    parser.add_argument('--train_type', type=str, default='', help='train type')
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')
    parser.add_argument('--full_params',action='store_true',help='full params update',default=False)
    parser.add_argument('--ckpt_file', type=str, default=None, help='checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='checkpoint directory')
    parser.add_argument('--ckpt_id', type=str, default=None, help='checkpoint id')
    # 添加DeepSpeed相关的参数
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to DeepSpeed config file')
    parser.add_argument('--deepspeed_stage', type=int, default=2, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO stage')
    parser.add_argument('--deepspeed_offload', action='store_true', help='Enable CPU offloading',default=False)
    parser.add_argument('--train_batch_size', type=int, default=None, help='train batch size')
    parser.add_argument('--world_size', type=int, help='world size')
    parser.add_argument('--local_rank', type=int, help='local rank')
    parser.add_argument('--stage', type=int, default=1,choices=[1,2,3], help='stage 1 only align attn output and stage 2 do kl-divergence,and stage 3 do SFT')
    parser.add_argument('--max_trained_tokens', type=int, default=100_000_000, help='max trained tokens')
    parser.add_argument('--terminate_at_loss', type=float, default=0, help='terminate the training at loss')

    parser.add_argument('--freeze_attention', type=int, default=0, help='Freeze Receptance,Key,Value')
    parser.add_argument('--use_bitsandbytes', type=int, default=0, help='apply 8bitlinear with bitsandbytes')
    parser.add_argument('--hybrid_attention_layers', type=int, default=8, help='Hybrid Attention Layers')
    parser.add_argument('--freeze_hybrid_attention', type=int, default=0, help='Freeze Hybrid Attention q,k,v')
    parser.add_argument('--allow_quant_frozen_layers', type=int, default=1, help='allow quant frozen layers')
    parser.add_argument('--quant_mode', type=str, default="int8", help='quant in peft mode except full,  can choose int8,nf4,none')
    parser.add_argument('--peftmode', type=str, default="full", help='peftmode full,lora,bone')
    parser.add_argument('--peft_r', type=int, default=32, help='peft block lora rank')
    parser.add_argument('--peft_scaling', type=float, default=0.5, help='peft block lora scaling')
    parser.add_argument('--peft_dropout', type=float, default=0.01, help='peft block lora dropout')

    parser.add_argument('--mlp_quant_mode', type=str, default="int8", help='MLP Quant mode int8,nf4')
    parser.add_argument('--bnb_optimizer_mode', type=int, default=1, help='Use Bitsandbytes 8bit optimizer AdamW:1 LION:2')
    #parser.add_argument('--deepspeed_lion_mode', type=int, default=0, help='Use Bitsandbytes 8bit optimizer AdamW')
    return parser

def lr_schedule(args, progress, step):
    if args.lr_final == args.lr_init: # or args.epoch_count == 0:
        lr = args.lr_init
    elif args.lr_final == 0 or args.lr_init == 0:  # linear decay
        lr = args.lr_init + (args.lr_final - args.lr_init) * progress
    else:  # exp decay
        lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))

    if step < args.warmup_steps:
        lr = lr * (0.01 + 0.99 * step / args.warmup_steps)
    
    return lr

def weight_decay_schedule(args, progress):
    if args.weight_decay_final > 0:
        return args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
    return args.weight_decay

def on_train_batch_start(args, model_engine, global_step, epoch):
    real_step = global_step + args.epoch_begin * args.epoch_steps
    max_epochs_trained_tokens = args.max_epochs * args.epoch_steps * args.global_bsz
    if args.max_trained_tokens > 0:
        max_trained_tokens = min(args.max_trained_tokens, max_epochs_trained_tokens)
    progress = (global_step - args.warmup_steps + 1) / (max_trained_tokens - args.warmup_steps)
    progress = min(1, max(0, progress))

    # LR schedule
    lr = lr_schedule(args, progress, real_step)
    
    # Weight decay schedule
    wd_now = weight_decay_schedule(args, progress)

    # 更新优化器参数
    for param_group in model_engine.optimizer.param_groups:
        if param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_now
        if args.layerwise_lr > 0:
            param_group["lr"] = lr * param_group["my_lr_scale"]
        else:
            param_group["lr"] = lr

    # 初始化日志（仅在第一步执行）
    if global_step == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "train_log.txt"), "a") as f:
            f.write(f"NEW RUN {time.strftime('%Y-%m-%d %H:%M:%S')}\n{vars(args)}\n")

    return lr, wd_now

# 在主训练循环开始前初始化tqdm
total_loss = 0
total_updates = 0
avg_loss = 0
def on_train_batch_end(args, batch_idx, model_engine, teacher_engine, loss,
                       teacher_loss, kl_loss, student_cross_entropy_loss,
                       global_step, epoch, last_log_time, token_per_step, 
                       is_accumulation_step, pbar, trained_tokens, grad_norm=0):
    current_time = time.time()
    elapsed_time = current_time - last_log_time
    steps_per_second = 1 / elapsed_time
    kt_s = token_per_step * steps_per_second / 1000  # K tokens per second
    global total_loss
    global total_updates
    global avg_loss
    total_loss += loss
    total_updates += 1
    avg_loss = total_loss / total_updates

    # 只在实际更新参数时更新进度条
    trained_tokens += token_per_step
    if is_accumulation_step and model_engine.global_rank == 0:
        if pbar is None:
            pbar = tqdm(total=args.epoch_steps, desc=f"Epoch {epoch}")

        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'steps/s': f'{steps_per_second:.2f}',
            'kt/s': f'{kt_s:.2f}',
            'trained_tokens': f'{trained_tokens / 1e6:.2f} MT',
            'remained_tokens': f'{(args.max_trained_tokens - trained_tokens) / 1e6:.2f} MT'
        })
        timer.print_stats(global_step)
        if args.wandb:
            wandb.log({
                "loss": loss,
                "lr": model_engine.optimizer.param_groups[0]['lr'],
                "grad_norm": grad_norm,
                "weight_decay": model_engine.optimizer.param_groups[0]['weight_decay'],
                "steps_per_second": steps_per_second,
                "kt/s": kt_s,
                "global_step": global_step,
                "Gtokens": global_step * token_per_step * args.accumulate_grad_batches / 1e9,
                "epoch": epoch,
                "teacher_loss": teacher_loss,
                "kl_loss": kl_loss,
                "student_cross_entropy_loss": student_cross_entropy_loss,
            })

    real_step = batch_idx
    if real_step % args.save_per_batches == 0 and real_step > 0:
        save_pth(args=args, model_engine=model_engine, epoch=epoch, real_step=real_step)

    return current_time, pbar, trained_tokens

def save_pth(args, model_engine, epoch, real_step):
        # 既存チェックポイントを整理（2世代残す）
        if os.path.exists(args.output_dir):
            if model_engine.local_rank == 0:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(args.output_dir, f))]
                checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                if len(checkpoints) > 2:
                    print(f'deleting older checkpoints {checkpoints[0]}')
                    import shutil
                    shutil.rmtree(os.path.join(args.output_dir, checkpoints[0]))

        output_dir = f"{args.output_dir}/epoch_{epoch}_step_{real_step}"
        print(f'saving checkpoint to {output_dir}')

        # 凍結されていないパラメータだけ保存
        if model_engine.global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = model_engine.module if hasattr(model_engine, "module") else model_engine

            full_state_dict = model_to_save.state_dict()
            trainable_keys = {name for name, param in model_to_save.named_parameters() if param.requires_grad}
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if k in trainable_keys}

            save_path = os.path.join(output_dir, "trainable_only_weights.pth")
            try:
                torch.save(filtered_state_dict, save_path)
                print(f"✅ Saved trainable-only weights to {save_path}")
            except Exception as e:
                print(f"Error saving weights: {e}")
                import traceback
                traceback.print_exc()


import torch.distributed as dist
def setup_distributed():
    dist.init_process_group(backend='rccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
import contextlib
from typing import List

class TeacherAttnManager:
    def __init__(self, model_engine, layers: List[int]):
        self.model_engine = model_engine
        self.layers = layers
        self.stored_teacher_attns = {}
        # self.stored_vfirst_state = {}
        # self.stored_kfirst_state = {}
        
    @contextlib.contextmanager
    def temporarily_remove_teacher_attn(self):
        """
        上下文管理器，临时移除所有层的teacher_attn,v_first_state并在退出时恢复
        """
        try:
            # 保存并移除所有teacher_attn
            for layer_idx in self.layers:
                attention_wrapper = self.model_engine.module.model.model.layers[layer_idx].self_attn
                if hasattr(attention_wrapper, 'teacher_attn'):
                    self.stored_teacher_attns[layer_idx] = attention_wrapper.teacher_attn
                    # 移除teacher_attn模块
                    if hasattr(attention_wrapper, '_modules') and 'teacher_attn' in attention_wrapper._modules:
                        del attention_wrapper._modules['teacher_attn']
                    attention_wrapper.teacher_attn = None
                # if hasattr(attention_wrapper, 'v_first_state'):
                #     self.stored_vfirst_state[layer_idx] = attention_wrapper.v_first_state
                #     attention_wrapper.v_first_state = None
                # if hasattr(attention_wrapper, 'k_first_state'):
                #     self.stored_kfirst_state[layer_idx] = attention_wrapper.k_first_state
                #     attention_wrapper.k_first_state = None
            
            yield  # 允许在此上下文中执行代码
            
        finally:
            # 恢复所有teacher_attn
            for layer_idx, stored_attn in self.stored_teacher_attns.items():
                attention_wrapper = self.model_engine.module.model.model.layers[layer_idx].self_attn
                attention_wrapper.teacher_attn = stored_attn
                # 重新注册为子模块
                if hasattr(attention_wrapper, 'add_module') and not hasattr(attention_wrapper, 'teacher_attn'):
                    attention_wrapper.add_module("teacher_attn", stored_attn)
                # v_first_state = self.stored_vfirst_state.get(layer_idx, None)
                # k_first_state = self.stored_kfirst_state.get(layer_idx, None)
                # if v_first_state is not None:
                #     attention_wrapper.v_first_state = v_first_state
                # if k_first_state is not None:
                #     attention_wrapper.k_first_state = k_first_state
            # 清空存储的引用
            self.stored_teacher_attns.clear()

import ctypes
def force_cpu_memory_cleanup():
    """
    CPUメモリを強制的にクリア
    """
    print("CPUメモリクリーンアップ開始...")
    
    # 1. PyTorchのガベージコレクション
    gc.collect()
    
    # 2. PyTorchの内部キャッシュをクリア
    torch.cuda.empty_cache()  # GPU側
    
    # 3. CPUメモリプールも解放を試行
    if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_emptyCache'):
        torch._C._cuda_emptyCache()
    
    # 4. より積極的なガベージコレクション
    for _ in range(3):
        collected = gc.collect()
        print(f"  GC回収: {collected} オブジェクト")
    
    # 5. 低レベルメモリ操作（Linux/Mac）
    try:
        libc = ctypes.CDLL("libc.so.6")  # Linux
        libc.malloc_trim(0)
        print("  malloc_trim実行完了")
    except:
        try:
            libc = ctypes.CDLL("libc.dylib")  # Mac
            libc.malloc_trim(0)
            print("  malloc_trim実行完了")
        except:
            print("  malloc_trim未対応")
    
    print("CPUメモリクリーンアップ完了")

def get_dataloaders(args, tokenizer):
    if args.preprocessed_data is not None:
        print(f'load preprocessed data from {args.preprocessed_data}')
        from data.multi_source_datasets import data_collator_with_pad 
        from functools import partial
        from torch.utils.data.distributed import DistributedSampler
        pad_token_id = tokenizer.pad_token_id
        data_collator = partial(data_collator_with_pad, max_seq_length=args.max_seq_length,pad_token_id=pad_token_id)
        
        # load all datasets
        train_datasets = []
        for data_path in args.preprocessed_data:  # 最后一个路径作为验证集
            ds = datasets.load_from_disk(data_path)
            train_datasets.append(ds)
        
        # merge all datasets
        train_ds = datasets.concatenate_datasets(train_datasets)
        
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.micro_bsz, 
            sampler=train_sampler,  # 使用分布式 sampler
            num_workers=1, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=data_collator
        )
        val_dataloader = None
        if args.local_rank == 0:
            print(f'load preprocessed data from {args.preprocessed_data} done')
    elif args.raw_data is not None:

        if len(args.raw_data) == 1:
            args.raw_data = args.raw_data[0].split(",")
        print(f'load raw data from {args.raw_data}')

        from data.raw_dataset import load_datasets_from_directories,TypedDataset,TypedStreamingCLMDataCollator
        all_ds,feature_types = load_datasets_from_directories(args.raw_data,tokenizer)
        typed_dataset = TypedDataset(all_ds, feature_types)
        # print(all_ds)
        # con_ds = datasets.concatenate_datasets(all_ds)
        # data_collator = StreamingCLMDataCollator(tokenizer=tokenizer, max_length=args.max_seq_length)
        data_collator = TypedStreamingCLMDataCollator(tokenizer=tokenizer, 
                                                  max_length=args.max_seq_length, 
                                                  min_length=args.max_seq_length, 
                                                  typed_dataset=typed_dataset,
                                                  need_to_pack=args.need_to_pack)
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            typed_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            typed_dataset, 
            batch_size=args.micro_bsz, 
            sampler=train_sampler,  # 使用分布式 sampler
            num_workers=4, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=data_collator
        ) 
        val_dataloader = None

    return train_dataloader, val_dataloader
