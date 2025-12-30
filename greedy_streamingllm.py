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
import json

from model.streaming_attention import StreamingAttention

# lm-eval imports (lazy loaded when needed)
lm_eval = None
HFLM = None
LM = None
StreamingHFLM = None

def load_lm_eval():
    global lm_eval, HFLM, LM, StreamingHFLM
    if lm_eval is None:
        import lm_eval as _lm_eval
        from lm_eval.models.huggingface import HFLM as _HFLM
        from lm_eval.api.model import LM as _LM
        lm_eval = _lm_eval
        HFLM = _HFLM
        LM = _LM

        # Define StreamingHFLM after HFLM is loaded
        class _StreamingHFLM(_HFLM):
            """Custom HFLM wrapper that properly initializes for wrapped models."""

            def __init__(
                self,
                model,
                tokenizer,
                batch_size=12,
                max_length=32768,
                device="cuda",
            ):
                # Initialize LM parent class directly (skip HFLM's complex init)
                _LM.__init__(self)

                # Store model and tokenizer
                self._model = model
                self.tokenizer = tokenizer
                self._config = model.config

                # Required HFLM attributes
                self._batch_size = int(batch_size)
                self._max_length = max_length
                self._device = torch.device(device) if isinstance(device, str) else device
                self.backend = "causal"
                self.trust_remote_code = True
                self.truncation = False

                # Token handling
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.vocab_size = self.tokenizer.vocab_size

                # Additional HFLM attributes that may be accessed
                self.add_bos_token = False
                self.custom_prefix_token_id = None
                self._prefix_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
                self.logits_cache = True
                self._rank = 0
                self._world_size = 1
                self.delta = None
                self.peft = None
                self.revision = "main"
                self.batch_schedule = 1
                self.batch_sizes = {}
                self.max_batch_size = 64

                # Additional attributes needed by HFLM methods
                self.mixed_precision_dtype = None
                self.softmax_dtype = None
                self.think_end_token = None
                self.chat_template_args = {}
                self.pretrained = None
                self.batch_size_per_gpu = batch_size
                self.accelerator = None

            @property
            def batch_size(self):
                return self._batch_size

            @property
            def max_length(self):
                return self._max_length

            @property
            def model(self):
                return self._model

            @property
            def device(self):
                return self._device

            @property
            def config(self):
                return self._config

            @property
            def eot_token_id(self):
                return self.tokenizer.eos_token_id

            @property
            def prefix_token_id(self):
                return self._prefix_token_id

            @property
            def max_gen_toks(self):
                return 256

            @property
            def rank(self):
                return self._rank

            @property
            def world_size(self):
                return self._world_size

            def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
                """Encode string to token ids."""
                if add_special_tokens is None:
                    add_special_tokens = False
                encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
                if left_truncate_len:
                    encoding = encoding[-left_truncate_len:]
                return encoding

            def tok_decode(self, tokens, skip_special_tokens=True):
                """Decode token ids to string."""
                return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

            def _model_generate(self, context, max_length, stop, **generation_kwargs):
                """Generate with proper handling of temperature and sampling kwargs."""
                # Handle temperature (copied from working wrapper)
                generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
                do_sample = generation_kwargs.get("do_sample", None)

                # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
                if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
                    generation_kwargs["do_sample"] = do_sample = False

                if do_sample is False and generation_kwargs.get("temperature") == 0.0:
                    generation_kwargs.pop("temperature")

                return self._model.generate(
                    input_ids=context,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    **generation_kwargs,
                )

        StreamingHFLM = _StreamingHFLM

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

def worker_process(local_rank:int, world_size:int, cli_config, *args, **kwargs):
    use_distributed = world_size > 1 and cli_config.eval_mode != 'lm_eval'

    try:
        if use_distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=local_rank,
                device_id=local_rank,
            )

        _worker_process(local_rank, world_size, cli_config, *args, **kwargs)
    except Exception as e:
        if local_rank == 0:
            #print(f"Error in worker: {e}")
            import traceback
            print(f"Error in worker\n", traceback.format_exc())

    if use_distributed:
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
    eval_mode:str = 'kl_div'
    tasks:list = field(default_factory=lambda: ['gsm8k'])
    output_dir:str = 'layer_eval_results'
    batch_size:int = 12
    limit:int|None = None  # limit samples per task (None = all)
    num_fewshot:int|None = None  # number of few-shot examples (None = task default)
    dtype:str = 'bfloat16'  # model dtype: bfloat16, float16, float32


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
    # Set name_or_path so lm-eval tasks (like ruler) can find the tokenizer
    model_config._name_or_path = cli_config.model_path

    # NOTE - entirely replacement attentions, and we will change the sliding window size as needed to simulate the original model
    model_config.layer_hybrid_types = ['radlads_replacement_attention'] * model_config.num_hidden_layers

    if local_rank == 0: print("instantiating customized model", cli_config.model_path)
    with init_empty_weights():
        model = StreamingHybridForCausalLM(model_config)

    if local_rank == 0: print("loading original model weights", cli_config.model_path)
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    torch_dtype = dtype_map.get(cli_config.dtype, torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(cli_config.model_path, device_map=device, torch_dtype=torch_dtype)
    base_weights = base_model.state_dict()
    # Copy generation_config from base model (not included in state_dict)
    model.generation_config = base_model.generation_config
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

    # ===== LM-EVAL MODE =====
    if cli_config.eval_mode == 'lm_eval':
        load_lm_eval()
        import csv
        import fcntl
        os.makedirs(cli_config.output_dir, exist_ok=True)

        layer_count = model_config.num_hidden_layers
        csv_path = os.path.join(cli_config.output_dir, "results.csv")

        # Create HFLM wrapper once (using custom wrapper that handles generation properly)
        lm = StreamingHFLM(model=model, tokenizer=tokenizer, batch_size=cli_config.batch_size, device=device)

        def extract_metrics(results):
            """Flatten lm-eval results dict into metric dict"""
            metrics = {}
            for task_name, task_results in results.items():
                for metric_name, value in task_results.items():
                    if metric_name != 'alias' and not metric_name.endswith('_stderr'):
                        # Convert numpy types to float
                        if hasattr(value, 'item'):
                            value = value.item()
                        metrics[f"{task_name}/{metric_name}"] = value
            return metrics

        def append_to_csv(layer_name, metrics):
            """Append a row to CSV with file locking for multi-GPU safety"""
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(csv_path) == 0:
                    # Write header
                    writer.writerow(['layer'] + list(metrics.keys()))
                writer.writerow([layer_name] + list(metrics.values()))
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Run baseline only on rank 0 (all layers with sliding_window=0 = full attention)
        if local_rank == 0:
            print(f"Running baseline evaluation (all full attention)...")
            for layer_id2 in range(layer_count):
                model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = 0

            baseline_results = lm_eval.simple_evaluate(
                model=lm,
                tasks=cli_config.tasks,
                limit=cli_config.limit,
                num_fewshot=cli_config.num_fewshot,
                random_seed=cli_config.seed,
                metadata={'tokenizer': cli_config.model_path},  # For ruler tasks
            )

            metrics = extract_metrics(baseline_results['results'])
            append_to_csv('baseline', metrics)
            print(f"Baseline: {metrics}")

        # Parallelize layer evaluation across GPUs (same pattern as KL divergence mode)
        # Tests: what if ONLY layer X uses SWA? (all others = full attention)
        for layer_id in range(local_rank, layer_count, world_size):
            print(f"[GPU {local_rank}] Evaluating layer {layer_id}/{layer_count-1} with SWA (others full attn)...")

            # Reset all to full attention
            for layer_id2 in range(layer_count):
                model.model.layers[layer_id2].self_attn.attn_replacement.sliding_window = 0

            # Enable SWA for this layer only
            model.model.layers[layer_id].self_attn.attn_replacement.sliding_window = cli_config.sliding_window_size

            # Run evaluation
            layer_results = lm_eval.simple_evaluate(
                model=lm,
                tasks=cli_config.tasks,
                limit=cli_config.limit,
                num_fewshot=cli_config.num_fewshot,
                random_seed=cli_config.seed,
                metadata={'tokenizer': cli_config.model_path},  # For ruler tasks
            )

            metrics = extract_metrics(layer_results['results'])
            append_to_csv(layer_id, metrics)
            print(f"[GPU {local_rank}] Layer {layer_id}: {metrics}")

        return

    # ===== KL DIVERGENCE MODE (original) =====
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
