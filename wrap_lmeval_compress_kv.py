from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from typing import Any, Dict, List, Optional, Tuple, Union
import torch, torch.nn as nn, torch.nn.functional as F
import inspect




from transformers import Cache, PretrainedConfig
from transformers.cache_utils import DynamicLayer

class CompressingCache(Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(layer_class_to_replicate=CompressingLayer, *args, **kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_kwargs['layer_idx'] = layer_idx

        return super().update(key_states, value_states, layer_idx, cache_kwargs)


compression_chunk_size = 256
n_window_chunks = 2

class CompressingLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.cumulative_length = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = super().update(key_states, value_states, cache_kwargs)
        B, H, T, D = keys.shape

        self.cumulative_length += key_states.shape[-2]

        layer_idx = cache_kwargs['layer_idx']

        if self.cumulative_length % compression_chunk_size == 0 and keys.shape[-2] > compression_chunk_size * n_window_chunks:
            num_top_k = 1024 - compression_chunk_size * n_window_chunks
            if num_top_k < self.cumulative_length - compression_chunk_size * n_window_chunks:
                keys_subset = keys[:, :, :-compression_chunk_size * n_window_chunks, :]
                keys_remaining = keys[:, :, -compression_chunk_size * n_window_chunks:, :]
                values_remaining = values[:, :, -compression_chunk_size * n_window_chunks:, :]
                # similarity
                c = keys_subset @ keys_subset.mT
                # mask diag so that self similarity of keys is not random based on length, since they're not normalized
                i = torch.arange(c.shape[-1], device=c.device)
                c[:, :, i, i] = float('nan')
                # Compute variance (manual calculation to handle NaN)
                mean = torch.nanmean(c, dim=-1, keepdim=True)
                variances = torch.nanmean((c - mean)**2, dim=-1)
                # Get top-k least similar keys (currently going to a total of cumulative_length ** 0.9 every chunk)
                #num_top_k = int((self.cumulative_length - compression_chunk_size * n_window_chunks) ** 0.9)
                if layer_idx == 0:
                    print("Compressing ", keys_subset.shape[2], " of ", self.cumulative_length, "to", num_top_k)
                top_k_indices = torch.topk(variances, num_top_k, dim=-1, largest=False).indices
                top_k_indices = top_k_indices.view(B, H, num_top_k, 1).expand(-1, -1, -1, D)
                # use only top-k least similar key indices as retained keys and values
                keys_subset = torch.gather(keys_subset, 2, top_k_indices)
                values_subset = torch.gather(values, 2, top_k_indices)
                keys = torch.cat([keys_subset, keys_remaining], dim=2)
                values = torch.cat([values_subset, values_remaining], dim=2)
                # if layer_idx == 0:
                #     print(f"Shape of top_k_keys: {keys.shape}")
                self.keys = keys
                self.values = values

        return keys, values

# for some reason replacing transformers.cache_utils.DynamicCache with CompressingCache didn't 'take' on gsm8k although it did for mmlu, so instead we patch GenerationMixin to replace the past_key_values
import transformers.generation
from transformers.generation.configuration_utils import GenerationConfig
class GenerationMixinReplacement(transformers.generation.GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ): # -> GenerateOutput | torch.LongTensor:
        # remove idiotic generation config settings from qwen
        kwargs['max_new_tokens'] = None
        # set up batched prefill
        kwargs['prefill_chunk_size'] = 256

        return super().generate(inputs, generation_config=generation_config, **kwargs)   

    # code from 5.0.0rc2 with bugfix for position_ids because 4.57.3 was broken for chunked prefill
    def _prefill(self, input_ids: torch.LongTensor, generation_config: GenerationConfig, model_kwargs):
        # force use of compressing cache
        if not isinstance(model_kwargs.get('past_key_values'), CompressingCache):
           model_kwargs['past_key_values'] = CompressingCache()

        if generation_config.prefill_chunk_size is None:
            model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            return self(**model_inputs, return_dict=True)
        else:  # Chunked prefill
            # Even if we are not compiling the forward, flex is always compiled when used. With chunked prefill, we may
            # end up needing just a bit more graphs than the default (which is 8). Doing this avoids very cryptic warnings
            torch._dynamo.config.cache_size_limit = 64

            chunk_size = generation_config.prefill_chunk_size
            input_chunks = torch.split(input_ids, chunk_size, dim=-1)

            if "past_key_values" not in model_kwargs:
                raise ValueError("Cannot use prefill chunking without a cache")

            model_forward = (
                self.get_compiled_call(generation_config.compile_config)
                if self._valid_auto_compile_criteria(model_kwargs, generation_config)
                else self.__call__
            )

            position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"

            attention_mask = model_kwargs.pop("attention_mask", None)
            past_length = 0
            for input_chunk in input_chunks:
                current_length = past_length + input_chunk.shape[-1]
                if attention_mask is not None:
                    model_kwargs["attention_mask"] = attention_mask[:, :current_length]
                model_kwargs["cache_position"] = torch.arange(
                    past_length, current_length, dtype=torch.long, device=input_chunk.device
                )

                # FIXME - there is some proper way to init position_ids here but I was unable to get it to work without just being an arange
                # if (
                #     attention_mask is not None
                #     and model_kwargs.get(position_ids_key) is None
                #     and position_ids_key in set(inspect.signature(model_forward).parameters.keys())
                # ):
                #     position_ids = attention_mask.long().cumsum(-1) - 1
                #     position_ids.masked_fill_(attention_mask == 0, 1)
                #     model_kwargs[position_ids_key] = position_ids
                #model_kwargs["position_ids"] = model_kwargs["cache_position"].unsqueeze(0) # this was HF's broken code that we removed

                model_inputs = self.prepare_inputs_for_generation(input_chunk, **model_kwargs)

                outputs = model_forward(**model_inputs, return_dict=True)

                model_kwargs["past_key_values"] = outputs.past_key_values
                past_length = current_length

            #model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
            model_kwargs["attention_mask"] = attention_mask
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
            # _ = model_kwargs.pop(position_ids_key, None)
            # Latest outputs contain next token logits
            return outputs
        
    # code from 5.0.0rc2 with bugfix for position_ids because 4.57.3 was broken for chunked prefill
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor,#: LogitsProcessorList,
        stopping_criteria,#: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer = None,#: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ): #-> GenerateNonBeamOutput | torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        prefill_consumed = False
        outputs = self._prefill(input_ids, generation_config, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            cache = None
            if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
                cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
                cache = model_kwargs[cache_key]
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=cache,
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=cache,
                )
        else:
            return input_ids
        
transformers.generation.GenerationMixin = GenerationMixinReplacement

# # Replace DynamicCache with our custom implementation (this works for non-generate evals)
# import transformers.cache_utils
# transformers.cache_utils.DynamicCache = CompressingCache

from lm_eval.__main__ import cli_evaluate # if we want to use lm-eval-harness
# from eval.eval import cli_evaluate # if we want to use evalchemy
if __name__ == '__main__':
    cli_evaluate()
