import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import Cache

from model.wrap_hf import create_model_class, create_config_class

from transformers.models.qwen2.modeling_qwen2 import repeat_kv

class SDPAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,

        use_cache: bool,
        hidden_states: torch.Tensor,
        position_embeddings,
        past_key_values,
        cache_position,
        **kwargs
    ):
        B, H, T, N = query.shape
        B, KVH, T, N = key.shape
        
        sdpa_kwargs = {}

        if H > KVH:
            key = repeat_kv(key, H // KVH)
            value = repeat_kv(value, H // KVH)
            #sdpa_kwargs = {"enable_gqa": True}

        is_causal = query.shape[2] > 1 and attention_mask is None
        if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
            is_causal = is_causal.item()

        attn_output = F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal, scale=scaling, **sdpa_kwargs)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output

class StreamingLLMAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,

        use_cache: bool,
        hidden_states: torch.Tensor,
        position_embeddings,
        past_key_values,
        cache_position,
        **kwargs
    ):
        q, k, v = query, key, value
        del query, key, value
        B, H, L, N = q.shape
        B, KVH, S, N = k.shape
        
        sdpa_kwargs = {}

        if H > KVH:
            # key = repeat_kv(key, H // KVH)
            # value = repeat_kv(value, H // KVH)
            sdpa_kwargs = {"enable_gqa": True}

        sliding_window = self.config.sliding_window_sizes[self.layer_idx]
        if sliding_window > 0:
            q_idx = torch.arange(S-L, S, device=q.device)[None, None, :, None]
            kv_idx = torch.arange(S, device=q.device)[None, None, None, :]
            window_mask = kv_idx >= q_idx - sliding_window
            if attention_mask is not None:
                assert attention_mask.dtype == torch.bool
                sink_indices = (S - attention_mask.view(B,L,S)[:,-1,:].sum(dim=-1)).view(B) # sink offset per batch idx
                sink_mask = kv_idx == sink_indices.view(B,1,1,1)
                #prefill_mha_mask = q_idx >= S - prefill_n_mha_tokens
                #attention_mask = attention_mask & (sink_mask | window_mask | prefill_mha_mask)
                attention_mask = attention_mask & (sink_mask | window_mask)
            else:
                sink_mask = kv_idx == 0
                attention_mask = kv_idx <= q_idx # causal
                attention_mask = attention_mask & (sink_mask | window_mask)

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, :S]

        is_causal = L > 1 and attention_mask is None
        if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
            is_causal = is_causal.item()

        attn_output = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal, scale=scaling, **sdpa_kwargs)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output

def my_create_config_class(parent_class):
    class MyStreamingLLMHybridConfig(parent_class):
        def __init__(self, sliding_window_sizes=None, **kwargs):
            super().__init__(**kwargs)      
            if sliding_window_sizes is None:
                sliding_window_sizes = [0] * self.num_hidden_layers
            self.sliding_window_sizes = sliding_window_sizes

    return MyStreamingLLMHybridConfig

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers import AutoConfig, PretrainedConfig
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import datasets
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
import os

@dataclass
class TokenizingCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int

    def __call__(self, examples: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
        texts = [example['text'] for example in examples]
        #print('[len(text) for text in texts]', [len(text) for text in texts])
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length + 1, # FIXME - fixed this from max_length without one added
            padding="max_length",
            return_tensors="pt",
            padding_side='right'
        )

        input_ids = tokenized["input_ids"][:, :-1]
        attention_mask = tokenized["attention_mask"][:, :-1]
        labels = tokenized["input_ids"][:, 1:].clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def worker_process(local_rank:int, world_size:int, *args, **kwargs):
    try:
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=local_rank,
                device_id=local_rank,
            )

        _worker_process(local_rank, world_size, *args, **kwargs)
    except Exception as e:
        if local_rank == 0:
            #print(f"Error in worker: {e}")
            import traceback
            print(f"Error in worker\n", traceback.format_exc())

    if world_size > 1:      
        dist.barrier()
        if local_rank == 0:
            print("Tearing down process group...")
        dist.destroy_process_group()

    if local_rank == 0:
        print("Done!")

@dataclass(kw_only=True)
class CLI_Config:
    num_gpus:int|None = None
    ctxlen:int = 2048
    micro_bsz:int = 4
    max_iters:int = 32
    dataset_name:str = "robbiegwaldd/dclm-10B"
    model_path:str = 'Qwen/Qwen2-0.5B-Instruct' # FIXME - use 3b or make all this stuff configurable
    base_model_class_path:str = 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'
    base_attention_class_path:str = 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'
    base_config_class_path:str = 'transformers.models.qwen2.configuration_qwen2.Qwen2Config'
    sliding_window_size:int = 256
    layer_hybrid_types:list|None = None
    seed:int = 1337


def _worker_process(local_rank:int, world_size:int, cli_config:CLI_Config):
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)
    #torch.set_default_device(device)

    tokenizer = AutoTokenizer.from_pretrained(cli_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(cli_config.model_path, trust_remote_code=True).to(device)

    StreamingLLMHybridForCausalLM = create_model_class(
        StreamingLLMAttention, 
        cli_config.base_model_class_path,
        cli_config.base_attention_class_path,
    )

    StreamingLLMHybridConfigParent = create_config_class(cli_config.base_config_class_path)
    StreamingLLMHybridConfig = my_create_config_class(StreamingLLMHybridConfigParent)

    if local_rank == 0: print("loading config", cli_config.model_path)
    # teacher_model_config = AutoConfig.from_pretrained(cli_config.model_path)
    # config_class = type(teacher_model_config)
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(cli_config.model_path, _from_auto=True)
    #config_dict['auto_map'] = {"AutoModelForCausalLM": "greedy_streamingllm.StreamingLLMHybridForCausalLM"}
    #if local_rank == 0: print(config_dict)
    model_config = StreamingLLMHybridConfig.from_dict(config_dict, **unused_kwargs)

    # NOTE - start out with entirely replacement attentions, so they get instantiated
    model_config.layer_hybrid_types = ['replacement_attention'] * model_config.num_hidden_layers

    #if local_rank == 0: print(model_config)
    if local_rank == 0: print("instantiating customized model", cli_config.model_path)
    #model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True) # trust_remote_code=True required for it to instantiate our class instead of the normal one
    model = StreamingLLMHybridForCausalLM(model_config)

    if local_rank == 0: print("loading original model weights", cli_config.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(cli_config.model_path, device_map='cpu')
    base_weights = base_model.state_dict()
    del base_model

    if local_rank == 0: print("copying original model weights", cli_config.model_path)
    model.load_state_dict(base_weights)
    del base_weights

    if local_rank == 0: print("moving model to device")
    model = model.to(device)
    model.eval()

    # print("running")
    # model.forward(input_ids=torch.zeros([1, 128], dtype=torch.long, device='cuda'), use_cache=False)
    # print("done")

    dataset = datasets.load_dataset(cli_config.dataset_name)['train'] #, streaming=True)

    if local_rank == 0: print(f"model: {cli_config.model_path} dataset: {cli_config.dataset_name}")
    if local_rank == 0: print(f"layer_id,kl_div_loss")

    sampler = None
    # if world_size > 1:
    #     sampler = DistributedSampler(
    #         dataset=dataset,
    #         num_replicas=world_size,
    #         rank=local_rank,
    #         shuffle=False, # NOTE - no shuffling
    #     )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cli_config.micro_bsz,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=TokenizingCollator(tokenizer, cli_config.ctxlen),
        sampler=sampler,
        shuffle=True,
    )

    # now that model is created, set which layers use the replacement to start
    model_config.layer_hybrid_types = ['full_attention'] * model_config.num_hidden_layers
    model_config.sliding_window_sizes = [0] * model_config.num_hidden_layers
    if cli_config.layer_hybrid_types is not None:
        for i, x in enumerate(cli_config.layer_hybrid_types):
            model_config.layer_hybrid_types[i] = 'replacement_attention' if x else 'full_attention'
            # Also set sliding window size for StreamingLLM layers
            if x:
                model_config.sliding_window_sizes[i] = cli_config.sliding_window_size

    with torch.no_grad():
        layer_count = len(model.model.layers)
        for layer_id in range(local_rank, layer_count, world_size):
            total_loss = torch.zeros([1], device=device)
            total_batchlen = 0
            torch.manual_seed(cli_config.seed)
            for step, data in enumerate(dataloader):
                if step >= cli_config.max_iters:
                    break

                input_ids = data['input_ids'].to(device)
                labels = data['labels'].to(device)
                attention_mask = data['attention_mask'].to(device=device, dtype=torch.bool)

                # run teacher model
                teacher_logits = model(input_ids).logits

                # change to student model with a single additional StreamingLLM layer
                old_sliding_window_size = model.config.sliding_window_sizes[layer_id]
                old_layer_hybrid_type = model.config.layer_hybrid_types[layer_id]
                model.config.sliding_window_sizes[layer_id] = cli_config.sliding_window_size
                model.config.layer_hybrid_types[layer_id] = 'replacement_attention'

                # run student model
                student_logits = model(input_ids).logits
                # change back to teacher model
                model.config.sliding_window_sizes[layer_id] = old_sliding_window_size
                model.config.layer_hybrid_types[layer_id] = old_layer_hybrid_type

                student_logits = student_logits.masked_fill_(~attention_mask.unsqueeze(-1), -9999999)
                teacher_logits = teacher_logits.masked_fill_(~attention_mask.unsqueeze(-1), -9999999)
               
                flat_attention_mask = attention_mask.view(-1)
                flat_student_logits = student_logits.view(-1, student_logits.size(-1))[flat_attention_mask]
                flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))[flat_attention_mask]
                flat_labels = labels.view(-1)[flat_attention_mask]
                # print('teacher ce', F.cross_entropy(flat_teacher_logits, flat_labels))
                # print('student ce', F.cross_entropy(flat_student_logits, flat_labels))

                #if local_rank == 0:
                #    print(f"Layer {layer_id} step {step} len {flat_student_logits.size(0)}")
                total_loss += F.kl_div(F.log_softmax(flat_student_logits, dim=-1), F.log_softmax(flat_teacher_logits, dim=-1), reduction='sum', log_target=True)
                total_batchlen += flat_student_logits.size(0)

            total_loss /= total_batchlen
            
            #if world_size > 1:
            #    dist.reduce(total_loss, 0, op = ReduceOp.AVG)
            # FIXME - save result
            print(f"{layer_id},{total_loss.item()}")

if __name__ == '__main__':
    import sys
    from config import parse_cmdline_configs
    cli_config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
    if errors != '':
        print(errors)
        exit(-1)

    if cli_config.num_gpus == 1:
        worker_process(0, 1, cli_config)
    else:
        # Required for CUDA multiprocessing
        mp.set_start_method('spawn', force=True) 

        if cli_config.num_gpus is None or cli_config.num_gpus == 0:
            cli_config.num_gpus = torch.cuda.device_count()

        if cli_config.num_gpus == 0:
            raise RuntimeError("No GPUs available!")

        # Use multiprocessing Manager for sharing results
        manager = mp.Manager()

        # Spawn processes for each GPU
        mp.spawn(
            worker_process,
            args=[cli_config.num_gpus, cli_config, ],
            nprocs=cli_config.num_gpus,
            join=False, #True
        )
