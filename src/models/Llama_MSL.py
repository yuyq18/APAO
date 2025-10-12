import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, Union
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import KwargsForCausalLM

from transformers.processing_utils import Unpack

class Llama_MSL(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--ce_tau", type=float, default=1.0, help="Temperature for cross-entropy loss")
        return parser

    def __init__(
        self, 
        config: LlamaConfig,
        args: Optional[object] = None,
    ):
        super().__init__(config)
        self.ce_tau = getattr(args, "ce_tau", 1.0)
        self.vocab_size = config.vocab_size

    def expand_past_key_values_for_negs(self, past_kv, repeat_factor: int):
        if hasattr(past_kv, "to_legacy_cache"):   # DynamicCache / StaticCache
            legacy = past_kv.to_legacy_cache()
        else:
            legacy = past_kv

        # 2) repeat_interleave on batch dim
        expanded_layers = []
        for k, v in legacy:
            k2 = k.repeat_interleave(repeat_factor, dim=0) #   k: [B, num_kv_heads, seq_len, head_dim]
            v2 = v.repeat_interleave(repeat_factor, dim=0) #   v: [B, num_kv_heads, seq_len, head_dim]
            expanded_layers.append((k2, v2))

        # Wrap into DynamicCache
        return_cache = DynamicCache.from_legacy_cache(tuple(expanded_layers))

        return return_cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # For inter-level, input_ids only include history, without targets
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,     # For inter-level, labels only include targets
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        constrain_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        history_outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        history_hidden_states = history_outputs.last_hidden_state  # (B, L, D)
        history_past_key_values = history_outputs.past_key_values

        if labels is not None:
            softmax_ce_loss = None
            loss = None

            B = input_ids.shape[0]
            T = labels.shape[1]
            repeat_factor = 1
            repeated_history_hidden_states = history_hidden_states.repeat_interleave(repeat_factor, dim=0)  # (B*1, L, D)
            repeated_attention_mask = attention_mask.repeat_interleave(repeat_factor, dim=0) # (B*1, L)
            attn_mask_full = torch.cat([repeated_attention_mask, torch.ones_like(labels, device=attention_mask.device)], dim=1) # (B*1, L+T)

            repeated_history_past_key_values = self.expand_past_key_values_for_negs(history_past_key_values, repeat_factor)

            # forward on labels
            outputs: BaseModelOutputWithPast = self.model(
                input_ids=labels,
                attention_mask=attn_mask_full,
                past_key_values=repeated_history_past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=None,
                **kwargs,
            )
            
            hidden_states = torch.cat([
                repeated_history_hidden_states[:, -1:, :],  # (B, 1, D)
                outputs.last_hidden_state[:, -T:-1, :]     # (B, T-1, D)
            ], dim=1) # (B, T, D)

            lm_logits = self.lm_head(hidden_states)  # (B, T, V)

            if constrain_mask is not None:
                lm_logits[constrain_mask == 0] = -float("inf")

            t_logits = lm_logits / self.ce_tau
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(t_logits.device)
            softmax_ce_loss = loss_fct(t_logits.reshape(-1, t_logits.size(-1)), labels.reshape(-1))

            # log
            self.softmax_ce_loss = softmax_ce_loss
            self.total_loss = softmax_ce_loss
            loss = softmax_ce_loss

            logits = lm_logits

        else: # label is None, inference / generate
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(history_hidden_states[:, slice_indices, :])
            if constrain_mask is not None:
                if isinstance(logits_to_keep, int):
                    constrain_mask = constrain_mask[:, -logits_to_keep:, :]
                logits[constrain_mask == 0] = -float("inf")
            
            logits = logits / self.ce_tau
            loss = None
            outputs = history_outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        constrain_mask=None,
        my_num_beams=20,
        my_prefix_allowed_tokens_fn=None,
        **kwargs
    ):
        if constrain_mask is not None:
            mask = constrain_mask[:, :input_ids.shape[1], :]
        elif my_prefix_allowed_tokens_fn is not None:
            mask = torch.zeros([input_ids.shape[0], input_ids.shape[1], self.vocab_size], dtype=torch.bool)
            for batch_id, beam_sent in enumerate(input_ids.view(-1, my_num_beams, input_ids.shape[-1])):
                for beam_id, sent in enumerate(beam_sent):
                    mask[batch_id * my_num_beams + beam_id, -1, my_prefix_allowed_tokens_fn(batch_id, sent)] = True
        else:
            mask = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return {
            "input_ids": model_inputs['input_ids'],
            "past_key_values": model_inputs['past_key_values'],
            "attention_mask": model_inputs['attention_mask'],
            "inputs_embeds": model_inputs['inputs_embeds'],
            "cache_position": model_inputs['cache_position'],
            "constrain_mask": mask,
            **kwargs,
        }
