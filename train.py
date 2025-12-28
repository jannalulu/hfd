from train_scripts.trainer import *

import sys
import os
import gc
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any, List

import argparse
import yaml
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PretrainedConfig

import json
from tqdm import tqdm
from train_scripts.profiler import timer

from logger import print0 as print

import dataclasses
from dataclasses import dataclass, field, fields, _HAS_DEFAULT_FACTORY
from contextlib import contextmanager

from config import parse_cmdline_configs

@dataclass(kw_only=True)
class TrainConfig:
    stage:int
    local_rank:int = 0
    teacher_hf:str = ''
    ckpt_file:str = ''
    output_dir:str = ''
    save_per_batches:int = 999999

    wandb:str = ''
    wandb_run_name:str = ''

    preprocessed_data:List|None = None
    raw_data:List|None = None
    need_to_pack:int = 1

    use_deepspeed:int = 1
    deepspeed_config:str = ''
    deepspeed_stage:int = 2
    deepspeed_offload:int = 0

    max_seq_length:int = 4096
    
    micro_bsz:int = 4
    n_gpu_per_node:int = 8
    n_nodes:int = 1
    
    global_bsz:int = -1 # FIXME - computed
    world_size:int = -1 # FIXME - computed

    epoch_begin:int = 0
    epoch_steps:int = -1 # FIXME - computed

    gradient_clip_val:float = 1.0
    grad_cp:int = 1
    accumulate_grad_batches:int = 1
    warmup_steps:int = 50
    lr_init:float = 1e-5
    lr_final:float = 1e-5
    weight_decay:float = 0.1
    weight_decay_final:float = -1

    layerwise_lr:int = 1

    max_trained_tokens:int = -1
    max_epochs:int = 1

    beta1:float = 0.9
    beta2:float = 0.98
    adam_eps:float = 1e-8

    kl_weight:float = 0.0
    ce_weight:float = 1.0

@dataclass(kw_only=True)
class ModelConfig:
    model_class_path:str = ''
    config_class_path:str = ''
    partial_config:dict
    # maybe we can specify a partial HF config in yaml format, and import the dict then merge with the teacher dict and save out the combined result
    # or even store that partial config here?

@dataclass(kw_only=True)
class CLI_Config:
    train:TrainConfig
    model:ModelConfig

def class_name_and_module_from_path(class_path):
    import importlib
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    target_class = getattr(module, class_name)
    return target_class, class_name, module

@contextmanager
def torch_default_dtype(dtype):
    tmp_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(tmp_dtype)

if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    while len(argv) > 0 and '=' in argv[0]:
        argv = argv[1:]
    cli_config, errors = parse_cmdline_configs(argv, CLI_Config)
    cli_config:CLI_Config
    if errors != '':
        print(errors)
        exit(-1)

    config_model = cli_config.model
    student_config_class, student_config_class_name, student_config_module = class_name_and_module_from_path(config_model.config_class_path)
    student_config_dict, unused_kwargs = PretrainedConfig.get_config_dict(cli_config.train.teacher_hf, _from_auto=True)
    student_config_dict = student_config_dict | cli_config.model.partial_config # dataclasses.asdict(cli_config.model.partial_config)
    student_config_dict['radlads_distillation_stage'] = cli_config.train.stage
    student_config = student_config_class.from_dict(student_config_dict, **unused_kwargs)
    # FIXME - need to somehow obtain the proper student_config.layer_hybrid_types

    config = cli_config.train

    if 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])
    config.world_size = config.n_nodes * config.n_gpu_per_node
    config.global_bsz = config.world_size * config.micro_bsz * config.accumulate_grad_batches

    if config.use_deepspeed:
        deepspeed.init_distributed()

    dtype = torch.bfloat16
    device = f'cuda:{config.local_rank}'

    # FIXME - get rid of this stuff somehow    
    os.environ["architecture"] = 'hxa079'
    # FIXME - this is incorrect when att_dim is larger
    os.environ["RWKV_HEAD"] = str(student_config_dict['num_attention_heads'])
    os.environ["RWKV_HEAD_SIZE_A"] = str(student_config_dict['head_dim'])
    os.environ["RWKV_MICRO_BSZ"] = str(config.micro_bsz)
    #os.environ['RWKV_ATTN_PEFTMODE'] = str(args.peftmode)
    #os.environ['RWKV_ATTN_QUANT'] = str(args.quant_mode)
    #os.environ['RWKV_ATTN_PEFT_R'] = str(args.peft_r)
    #os.environ['RWKV_ATTN_PEFT_SCALING'] = str(args.peft_scaling)
    #os.environ['RWKV_ATTN_PEFT_DROPOUT'] = str(args.peft_dropout)

    student_model_class, student_model_class_name, student_model_module = class_name_and_module_from_path(config_model.model_class_path)

    if config.stage == 1:
        # NOTE - oh right this is only ZeRO stages 1,2 which do not partition model weights so we can fit them on GPU by definition
        # could still be slow to load on each process but hey

        original_model = AutoModelForCausalLM.from_pretrained(config.teacher_hf, dtype=dtype, device_map='cpu', trust_remote_code=True)
        original_weights = original_model.state_dict()
        del original_model
        teacher_model = None

        keys = list(original_weights.keys())
        for k in keys:
            # this gives the inline teacher the copies it needs of the self_attn weights
            if '.self_attn.' in k:
                original_weights[k.replace('.self_attn.', '.self_attn.teacher_attn.')] = original_weights[k].clone() # FIXME - clone?

            # FIXME - this is really somewhat architecture specific (e.g. Qwen and Llama work but Deepseek won't)
            # scale the student weights we want to
            if '.self_attn.q_proj.weight' in k or '.self_attn.k_proj.weight' in k or '.self_attn.o_proj.weight' in k:
                original_weights[k] *= 0.5
            if '.self_attn.v_proj.weight' in k:
                original_weights[k] *= 0.3

        # NOTE - must not create with empty weights, but could create on device
        # FIXME - actually, if our student supports reset_parameters() then we could start empty or start on device but skip reset_parameters() since we're going to be loading the weights anyway... 
        #  this would be faster to instantiate
        with torch_default_dtype(dtype), torch.device(device):
            student_model = student_model_class(student_config)

        # FIXME - should probably detect error when we load weights that don't end up in use
        student_model.load_state_dict(original_weights, strict=False)

    elif config.stage == 2:
        teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_hf, dtype=dtype, device_map=device, trust_remote_code=True)
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        teacher_weights = teacher_model.state_dict()
       
        weights = torch.load(config.ckpt_file, map_location='cpu', mmap=True)
        # merge in teacher weights that were not in the student (things that weren't saved due to not being trained, e.g. FFNs)
        # FIXME - we are going to want to allow selective layers to remain attention here, and import those... so that stage 1 can train all the layers without requiring stage 2 to use them all
        # FIXME - also allow NoPE layers to be added here - do we want to maybe train those separately too in their own stage 1 to copy from?
        for n, p in teacher_weights.items():
            if n not in weights:
                weights[n] = p.to('cpu')

        #student_model = AutoModelForCausalLM.from_pretrained(config.teacher_hf, dtype=dtype, device_map=config.local_rank, trust_remote_code=True)
        # FIXME - consider saving disk space by saving only unfrozen weights and reloading teacher first, then replacing with saved student weights
        from accelerate import init_empty_weights
        with torch_default_dtype(dtype), init_empty_weights():
            student_model = student_model_class(student_config)
        assert len(set(student_model.state_dict().keys()) - set(weights.keys())) == 0, "Student model had parameters that were not present in the loaded checkpoint"
        student_model.load_state_dict(weights, strict=False, assign=True)
        del weights
    else:
        assert False, f"distillation stage {config.stage} not supported"

    # # FIXME - this doesn't appear to work properly in HF yet, but neither does our inner checkpointing in the attention replacement
    # if config.grad_cp:
    #     student_model.gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})

    # freeze everything except the new attention replacements
    student_model.requires_grad_(False)
    #student_model.get_input_embeddings().requires_grad_(True) # FIXME - should we unfreeze the embeddings since those don't usually work properly frozen with deepspeed? or is this only in zero_stage3?
    for layer in student_model.model.layers:
        layer.self_attn.requires_grad_(True)
        if layer.self_attn.teacher_attn is not None:
            layer.self_attn.teacher_attn.requires_grad_(False)

    # for n, p in student_model.named_parameters():
    #     print(n, p.requires_grad)
    
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, val_dataloader = get_dataloaders(config, tokenizer)

    from train_scripts.train_functions import configure_optimizer, configure_optimizer_stage2, train_step

    if config.stage == 1:
        optimizer = configure_optimizer(student_model, config)
    elif config.stage == 2:
        optimizer = configure_optimizer_stage2(student_model, config)

    assert config.use_deepspeed
    if config.use_deepspeed:
        if config.deepspeed_config:
            with open(config.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            ds_config = {
                "zero_force_ds_cpu_optimizer": True,
                "distributed_backend": "rccl",
                "train_batch_size": config.global_bsz,
                "bf16": {
                    "enabled": True
                },
                # "torch_autocast": {
                #     "enabled": True,
                #     "dtype": "bfloat16",
                # },
                "fp32_reduce_scatter": True,
                "zero_optimization": {
                    "stage": config.deepspeed_stage,

                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": False,
                        "buffer_count": 4,
                        'ratio':1.0
                    },

                    "allgather_partitions": True,
                    "sub_group_size": 1e7,
                    "overlap_comm": True,
                    #"contiguous_gradients": False
                },
                "gradient_clipping": config.gradient_clip_val,
                #"gradient_checkpointing": config.grad_cp == 1, # FIXME - does this really do anything?
                "zero_force_ds_cpu_initialization": True,
                "zero_allow_untested_optimizer": True,
                "gradient_accumulation_steps": config.accumulate_grad_batches if config.accumulate_grad_batches > 1 else None,
                "wall_clock_breakdown": False,
                #"dump_state": True
            }
        if not config.deepspeed_offload:
            ds_config['zero_optimization']['offload_optimizer'] = None
            ds_config['zero_optimization']['offload_param'] = None
            ds_config['zero_force_ds_cpu_optimizer'] = False
            ds_config['zero_force_ds_cpu_initialization'] = False

        student_model_engine, optimizer, _, _ = deepspeed.initialize(
            model=student_model,  
            model_parameters=(p for p in student_model.parameters() if p.requires_grad),
            optimizer=optimizer,
            config=ds_config
        )
        del student_model

    teacher_engine = teacher_model

    @contextmanager
    def temporarily_remove_teacher_attn(student_model_engine):
        stored_teacher_attns = {}

        try:
            for layer_idx, layer in enumerate(student_model_engine.module.model.layers):
                attention_wrapper = layer.self_attn
                if hasattr(attention_wrapper, 'teacher_attn'):
                    stored_teacher_attns[layer_idx] = attention_wrapper.teacher_attn
                    if hasattr(attention_wrapper, '_modules') and 'teacher_attn' in attention_wrapper._modules:
                        del attention_wrapper._modules['teacher_attn']
                    attention_wrapper.teacher_attn = None
            
            yield
            
        finally:
            for layer_idx, stored_attn in stored_teacher_attns.items():
                attention_wrapper = student_model_engine.module.model.layers[layer_idx].self_attn
                attention_wrapper.teacher_attn = stored_attn
                if hasattr(attention_wrapper, 'add_module') and not hasattr(attention_wrapper, 'teacher_attn'):
                    attention_wrapper.add_module("teacher_attn", stored_attn)
            stored_teacher_attns.clear()

    config.epoch_steps = len(train_dataloader) // (config.accumulate_grad_batches)
    global_step = 0
    last_log_time = time.time()
    token_per_step = config.max_seq_length * config.micro_bsz * config.world_size
    terminate = False
    pbar = None
    trained_tokens = 0

    if config.wandb and student_model_engine.global_rank == 0:
        print(f'init wandb, project is {config.wandb}, name is {config.wandb_run_name}')
        wandb.init(project=config.wandb, name=config.wandb_run_name, config=config)
        print(f'begin training with {config.max_epochs} epochs, {config.max_trained_tokens} max_trained_tokens, {config.epoch_steps} epoch_steps, {token_per_step} token_per_step, {len(train_dataloader)} dataloader_len')

    for epoch in range(config.max_epochs):
        if terminate:
            break

        student_model_engine.train()
        if student_model_engine.global_rank == 0:
            pbar = tqdm(total=config.epoch_steps, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(train_dataloader):
            
            lr, wd_now = on_train_batch_start(config, student_model_engine, global_step, epoch)

            batch = {k: v.to(student_model_engine.device) for k, v in batch.items()}
            
            loss, teacher_loss, kl_loss, student_cross_entropy_loss = train_step(student_model_engine, batch, config, teacher_engine, tokenizer)
            
            student_model_engine.backward(loss)

            is_accumulation_step = (batch_idx + 1) % config.accumulate_grad_batches == 0
            grad_norm = None
            if is_accumulation_step:
                global_step += 1 
                try:
                    grad_norm = student_model_engine.get_global_grad_norm()
                except AttributeError:
                    grad_norm = None
               
            student_model_engine.step()

            last_log_time, pbar, trained_tokens = on_train_batch_end(
                config, batch_idx, student_model_engine,teacher_engine, loss.item(), teacher_loss, kl_loss, student_cross_entropy_loss,
                global_step, epoch, last_log_time, token_per_step, is_accumulation_step, pbar, trained_tokens, grad_norm=grad_norm
            )

            if trained_tokens >= config.max_trained_tokens:
                terminate = True
                break

        # save at end of epoch or end of training
        if config.output_dir:
            save_pth(config, student_model_engine, epoch, batch_idx)

        # if config.output_dir:
        #     if config.use_deepspeed:
                
        #         with temporarily_remove_teacher_attn(student_model_engine):
        #             try:
        #                 print(f"Saving checkpoint to {config.output_dir} at epoch {epoch} rank {student_model_engine.global_rank}")
        #                 student_model_engine.save_checkpoint(config.output_dir, f"checkpoint-epoch{epoch}",exclude_frozen_parameters=True)
        #             except Exception as e:
        #                 print(f"Error saving checkpoint: {e}")
        #                 import traceback
        #                 traceback.print_exc()
    
    print("Done!")
