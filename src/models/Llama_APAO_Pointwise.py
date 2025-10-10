import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import Optional, Union
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import KwargsForCausalLM

from transformers.processing_utils import Unpack

class Llama_APAO_Pointwise(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--beta", type=float, default=0.1, help="Weight for prefix loss")
        parser.add_argument("--ce_tau", type=float, default=1.0, help="Temperature for cross-entropy loss")
        parser.add_argument("--prefix_tau", type=float, default=1.0, help="Temperature for prefix loss")
        parser.add_argument("--eta_adapt", type=float, default=0.0, help="Eta for adaptive loss weight")
        parser.add_argument("--wo_ce_loss", action='store_true', default=False, help="Removing cross-entropy loss")
        return parser

    def __init__(
        self, 
        config: LlamaConfig,
        args: Optional[object] = None,
    ):
        super().__init__(config)
        self.beta = getattr(args, "beta", 0.1)
        self.ce_tau = getattr(args, "ce_tau", 1.0)
        self.prefix_tau = getattr(args, "prefix_tau", 1.0)
        self.eta_adapt = getattr(args, "eta_adapt", 0.0)
        self.wo_ce_loss = getattr(args, "wo_ce_loss", False)

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
            prefix_losses = []
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

            t_logits = lm_logits / self.ce_tau
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(t_logits.device)
            softmax_ce_loss = loss_fct(t_logits.reshape(-1, t_logits.size(-1)), labels.reshape(-1))

            log_probs = F.log_softmax(t_logits, dim=-1) # (B, T, V)
            token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) # (B, T)
            # Prefix-level Pointwise Loss
            for t in range(1, T):  # prefix t = 1,2,3,4
                prefix_logp = (token_logp[:, :t]).sum(dim=1) / t  # (B,)
                prefix_logp = prefix_logp / self.prefix_tau
                prefix_loss_t = (-prefix_logp).mean()  # (B,)
                prefix_losses.append(prefix_loss_t)

            # Adaptive Loss Weight
            all_prefix_losses = torch.stack(prefix_losses)
            all_prefix_losses = torch.nan_to_num(all_prefix_losses, nan=0.0, posinf=0.0, neginf=0.0)
            base_losses = all_prefix_losses.detach()

            eps = 1e-12
            # Step 1: normalize
            denom1 = base_losses.sum()
            if denom1.abs() < eps:
                w = torch.full_like(base_losses, 1.0 / base_losses.numel())
            else:
                w = base_losses / (denom1 + eps)
            # Step 2: w <- w * exp(eta * L)
            eta = getattr(self, "eta_adapt", 0.0)
            w = w * torch.exp(eta * base_losses)
            # Step 3: normalize
            w = w / (w.sum() + eps)
            # Final weighted loss
            weighted_prefix_loss = (w * all_prefix_losses).sum()
            self.adaptive_weights = w.detach().cpu().tolist()

            # log
            self.softmax_ce_loss = softmax_ce_loss
            self.prefix_losses = prefix_losses
            self.weighted_prefix_loss = self.beta * weighted_prefix_loss

            if self.wo_ce_loss:
                self.total_loss = self.beta * weighted_prefix_loss
                loss = self.beta * weighted_prefix_loss
            else:
                self.total_loss = softmax_ce_loss + self.beta * weighted_prefix_loss
                loss = softmax_ce_loss + self.beta * weighted_prefix_loss

            logits = lm_logits

        else: # label is None, inference / generate
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(history_hidden_states[:, slice_indices, :])
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
