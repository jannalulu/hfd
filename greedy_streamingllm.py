import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import Cache

from model.wrap_hf import create_model_class, create_config_class

from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from accelerate import init_empty_weights

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

from model.streaming_attention import StreamingAttention

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
    model_path:str = 'Qwen/Qwen2-3B-Instruct'
    base_model_class_path:str = 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'
    base_attention_class_path:str = 'transformers.models.qwen2.modeling_qwen2.Qwen2Attention'
    base_config_class_path:str = 'transformers.models.qwen2.configuration_qwen2.Qwen2Config'
    sliding_window_size:int = 256
    swa_layer_ids:list = field(default_factory=list)
    seed:int = 1337
    iterate:int = 1
    test:int = 0


def _worker_process(local_rank:int, world_size:int, cli_config:CLI_Config):
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)
    #torch.set_default_device(device)

    tokenizer = AutoTokenizer.from_pretrained(cli_config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    StreamingHybridForCausalLM = create_model_class(
        StreamingAttention, 
        cli_config.base_model_class_path,
        cli_config.base_attention_class_path,
    )

    StreamingHybridConfigParent = create_config_class(cli_config.base_config_class_path)
    class StreamingHybridConfig(StreamingHybridConfigParent):
        def __init__(self, streaming_sliding_window:int|None = None, **kwargs):
            super().__init__(**kwargs)
            self.streaming_sliding_window = streaming_sliding_window

    if local_rank == 0: print("loading config", cli_config.model_path)
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(cli_config.model_path, _from_auto=True)
    model_config = StreamingHybridConfig.from_dict(config_dict, **unused_kwargs)

    # NOTE - entirely replacement attentions, and we will change the sliding window size as needed to simulate the original model
    model_config.layer_hybrid_types = ['radlads_replacement_attention'] * model_config.num_hidden_layers

    if local_rank == 0: print("instantiating customized model", cli_config.model_path)
    with init_empty_weights():
        model = StreamingHybridForCausalLM(model_config)

    if local_rank == 0: print("loading original model weights", cli_config.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(cli_config.model_path, device_map=device)
    base_weights = base_model.state_dict()
    del base_model

    if local_rank == 0: print("moving original model weights", cli_config.model_path)
    model.load_state_dict(base_weights, assign=True)
    del base_weights

    model.eval()

    if cli_config.test:
        model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs)
        print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        return

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

    with torch.no_grad():
        layer_count = len(model.model.layers)
        for layer_id in range(local_rank, layer_count, world_size if cli_config.iterate else 999999):
            if cli_config.iterate and layer_id in cli_config.swa_layer_ids:
                continue
            total_loss = torch.zeros([1], device=device)
            total_batchlen = 0
            torch.manual_seed(cli_config.seed)
            for step, data in enumerate(dataloader):
                if step >= cli_config.max_iters:
                    break

                input_ids = data['input_ids'].to(device)
                # labels = data['labels'].to(device)
                attention_mask = data['attention_mask'].to(device=device, dtype=torch.bool)

                # change to teacher model with all attention
                for layer_id2 in range(model_config.num_hidden_layers):
                    model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = 0
                # add swa layers to teacher model
                # for layer_id2 in cli_config.swa_layer_ids:
                #     model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = cli_config.sliding_window_size

                # run teacher model
                teacher_logits = model(input_ids).logits

                # change to student model with a single additional swa layer
                for layer_id2 in cli_config.swa_layer_ids:
                    model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = cli_config.sliding_window_size
                if cli_config.iterate:
                    model.model.layers[layer_id].self_attn.attn_replacement.sliding_window = cli_config.sliding_window_size

                # run student model
                student_logits = model(input_ids).logits

                # student_logits = student_logits.masked_fill_(~attention_mask.unsqueeze(-1), -9999999)
                # teacher_logits = teacher_logits.masked_fill_(~attention_mask.unsqueeze(-1), -9999999)
               
                flat_attention_mask = attention_mask.view(-1)
                flat_student_logits = student_logits.view(-1, student_logits.size(-1))[flat_attention_mask]
                flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))[flat_attention_mask]
                # flat_labels = labels.view(-1)[flat_attention_mask]
                # print('teacher ce', F.cross_entropy(flat_teacher_logits, flat_labels))
                # print('student ce', F.cross_entropy(flat_student_logits, flat_labels))

                #if local_rank == 0:
                #    print(f"Layer {layer_id} step {step} len {flat_student_logits.size(0)}")
                chunk_size = 256
                for i in range(0, flat_student_logits.shape[0], 256):
                    total_loss += F.kl_div(F.log_softmax(flat_student_logits[i:i+chunk_size], dim=-1), F.log_softmax(flat_teacher_logits[i:i+chunk_size], dim=-1), reduction='sum', log_target=True)
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
