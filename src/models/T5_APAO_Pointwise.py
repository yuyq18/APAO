import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration, __HEAD_MASK_WARNING_MSG
)
import warnings
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

class T5_APAO_Pointwise(T5ForConditionalGeneration):

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
        config: T5Config,
        args: Optional[object] = None,
    ):
        super().__init__(config)
        self.beta = getattr(args, "beta", 0.1)
        self.ce_tau = getattr(args, "ce_tau", 1.0)
        self.prefix_tau = getattr(args, "prefix_tau", 1.0)
        self.eta_adapt = getattr(args, "eta_adapt", 0.0)
        self.wo_ce_loss = getattr(args, "wo_ce_loss", False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        if labels is not None:

            T = labels.shape[1]

            t_logits = lm_logits / self.ce_tau
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(t_logits.device)
            softmax_ce_loss = loss_fct(t_logits.reshape(-1, t_logits.size(-1)), labels.reshape(-1))

            # Prefix-level Pointwise Loss
            prefix_losses = []
            log_probs = F.log_softmax(t_logits, dim=-1) # (B, T, V)
            token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1) # (B, T)

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

        else:
            lm_logits = lm_logits
            loss = None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
