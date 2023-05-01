import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import softmax_backward_data
from torch.utils import checkpoint

from configuration_nort5 import NorT5Config
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import gelu_new
from transformers.modeling_outputs import (
    Seq2SeqModelOutput, Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
)


class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.main_input_name = "input_ids"

        self.relative_embedding = RelativeEmbedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing
    
    def forward(self, hidden_states, attention_mask):
        relative_embedding = self.relative_embedding()
        hidden_states, attention_probs = [hidden_states], []

        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_state, attention_p = checkpoint.checkpoint(layer, hidden_states[-1], attention_mask, relative_embedding)
            else:
                hidden_state, attention_p = layer(hidden_states[-1], attention_mask, relative_embedding)

            hidden_states.append(hidden_state)
            attention_probs.append(attention_p)

        return hidden_states, attention_probs


class Decoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.self_relative_embedding = RelativeEmbedding(config)
        self.cross_relative_embedding = RelativeEmbedding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[1].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing

    def forward(self, x, encoder_output, encoder_padding_mask, past_key_values=None):
        self_relative_embedding = self.self_relative_embedding()
        cross_relative_embedding = self.cross_relative_embedding()

        if past_key_values is None:
            autoreg_mask = torch.triu(
                torch.full((x.size(0), x.size(0)), True, device=x.device),
                diagonal=1
            )
        else:
            autoreg_mask = None

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        hidden_states, self_attention_probs, cross_attention_probs, key_value_states = [x], [], [], []
        for layer, past_key_value in zip(self.layers, past_key_values):
            if self.activation_checkpointing:
                hidden_state, self_attention_p, cross_attention_p, key_value_state = checkpoint.checkpoint(layer, hidden_states[-1], autoreg_mask, encoder_output, encoder_padding_mask, self_relative_embedding, cross_relative_embedding, past_key_value=None)
            else:
                hidden_state, self_attention_p, cross_attention_p, key_value_state = layer(hidden_states[-1], autoreg_mask, encoder_output, encoder_padding_mask, self_relative_embedding, cross_relative_embedding, past_key_value=past_key_value)

            hidden_states.append(hidden_state)
            self_attention_probs.append(self_attention_p)
            cross_attention_probs.append(cross_attention_p)
            key_value_states.append(key_value_state)

        return hidden_states, self_attention_probs, cross_attention_probs, key_value_states


class MaskClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config, is_cross_attention=False)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask, relative_embedding):
        attention_output, attention_probs, _ = self.attention(x, x, padding_mask, relative_embedding)
        x = x + attention_output
        x = x + self.mlp(x)
        return x, attention_probs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config, is_cross_attention=False)
        self.cross_attention = Attention(config, is_cross_attention=True)
        self.mlp = FeedForward(config)

    def forward(self, x, autoreg_mask, encoder_output, encoder_padding_mask, self_relative_embedding, cross_relative_embedding, past_key_value=None):
        query_offset = 0
        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
            query_offset = self_attn_past_key_value[0].size(2)
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        x_, self_attention_probs, self_key_value_state = self.self_attention(x, x, autoreg_mask, self_relative_embedding, past_key_value=self_attn_past_key_value, query_offset=query_offset)
        x = x + x_
        x_, cross_attention_probs, cross_key_value_state = self.cross_attention(x, encoder_output, encoder_padding_mask, cross_relative_embedding, past_key_value=cross_attn_past_key_value, query_offset=query_offset)
        x = x + x_
        x = x + self.mlp(x)

        return x, self_attention_probs, cross_attention_probs, self_key_value_state + cross_key_value_state


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * gelu_new(gate)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        if mask is not None:
            x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        if mask is not None:
            x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        input_grad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return input_grad, None, None


class Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        self.config = config
        self.is_cross_attention = is_cross_attention

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_q = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.in_proj_k = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        position_indices = torch.arange(512, dtype=torch.long).unsqueeze(1) \
            - torch.arange(512, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, 512)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_q.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_k.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_q.bias.data.zero_()
        self.in_proj_k.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, q, kv, attention_mask, relative_embedding, past_key_value=None, query_offset=0):
        key_len, batch_size, _ = kv.size()
        query_len, _, _ = q.size()

        if not self.is_cross_attention or past_key_value is None or past_key_value[0].size(1) != kv.size(0):
            kv = self.pre_layer_norm(kv)
            key = self.in_proj_k(kv)  # shape: [T, B, D]
            value = self.in_proj_v(kv)  # shape: [T, B, D]
            key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)  # shape: [BxH, T, D]
            value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)  # shape: [BxH, T, D]

        if past_key_value is not None:
            if not self.is_cross_attention:
                key = torch.cat([past_key_value[0].flatten(0, 1), key], dim=1)
                value = torch.cat([past_key_value[1].flatten(0, 1), value], dim=1)
                key_len = key.size(1)
            elif past_key_value[0].size(1) == kv.size(0):
                key = past_key_value[0].flatten(0, 1)
                value = past_key_value[1].flatten(0, 1)

        if self.position_indices.size(0) < max(query_len, key_len):
            position_indices = torch.arange(max(query_len, key_len), dtype=torch.long).unsqueeze(1) \
                - torch.arange(max(query_len, key_len), dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 512)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(q.device), persistent=True)

        q = self.pre_layer_norm(q)
        query = self.in_proj_q(q)  # shape: [T, B, D]
        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)
        
        query_pos = self.in_proj_q(self.dropout(relative_embedding))  # shape: [2T-1, D]
        query_pos = query_pos.view(-1, self.num_heads, self.head_size)  # shape: [2T-1, H, D]
        key_pos = self.in_proj_k(self.dropout(relative_embedding))  # shape: [2T-1, D]
        key_pos = key_pos.view(-1, self.num_heads, self.head_size)  # shape: [2T-1, H, D]

        query_ = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key_ = key.view(batch_size, self.num_heads, key_len, self.head_size)
        
        attention_c_p = torch.einsum("bhqd,khd->bhqk", query_, key_pos.squeeze(1) * self.scale)
        attention_p_c = torch.einsum("bhkd,qhd->bhqk", key_ * self.scale, query_pos.squeeze(1))
        position_indices = self.position_indices[query_offset:query_offset+query_len, :key_len].expand(batch_size, self.num_heads, -1, -1)
        attention_c_p = attention_c_p.gather(3, position_indices)
        attention_p_c = attention_p_c.gather(2, position_indices)
        
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        attention_scores.add_(attention_c_p)
        attention_scores.add_(attention_p_c)

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)

        key = key.detach().unflatten(0, (-1, self.num_heads))
        value = value.detach().unflatten(0, (-1, self.num_heads))

        return context, attention_probs.detach(), (key, value)


class WordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        return self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))


class RelativeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self):
        return self.relative_layer_norm(self.relative_embedding)


#
# HuggingFace wrappers
#

class NorT5PreTrainedModel(PreTrainedModel):
    config_class = NorT5Config
    base_model_prefix = "norT5"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Encoder):
            module.activation_checkpointing = value

    def _init_weights(self, module):
        pass  # everything is already initialized


class NorT5Model(NorT5PreTrainedModel):
    def __init__(self, config, add_lm_layer=False, add_decoder=True):
        super().__init__(config)
        self.config = config

        self.cls_token_id = config.cls_token_id
        self.sep_token_id = config.sep_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

        self.embedding = WordEmbedding(config)
        self.encoder = Encoder(config, activation_checkpointing=False)
        self.decoder = Decoder(config, activation_checkpointing=False) if add_decoder else None
        self.classifier = MaskClassifier(config) if add_lm_layer else None

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def get_encoder(self):
        return self.get_encoder_output

    def get_decoder(self):
        return self.get_decoder_output

    def set_decoder_special_tokens(self, target_id):
        target_id.masked_fill_(target_id == self.cls_token_id, self.bos_token_id)
        target_id.masked_fill_(target_id == self.sep_token_id, self.eos_token_id)
        return target_id

    def _shift_right(self, input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.bos_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.pad_token_id)

        return shifted_input_ids

    def get_encoder_output(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict = False
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        else:
            attention_mask = ~attention_mask.bool()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        static_embeddings = self.embedding(input_ids.t())
        contextualized_embeddings, attention_probs = self.encoder(static_embeddings, attention_mask)
        contextualized_embeddings = [e.transpose(0, 1) for e in contextualized_embeddings]
        last_layer = contextualized_embeddings[-1]
        contextualized_embeddings = [contextualized_embeddings[0]] + [
            contextualized_embeddings[i] - contextualized_embeddings[i - 1]
            for i in range(1, len(contextualized_embeddings))
        ]

        if not return_dict:
            return (
                last_layer,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
        
        return BaseModelOutput(
            last_hidden_state=last_layer,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )
    
    def get_decoder_output(
        self,
        target_ids: torch.Tensor = None,
        encoder_output: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict = False
    ):
        batch_size, seq_length, _ = encoder_output.shape
        device = target_ids.device

        if attention_mask is None:
            attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        else:
            attention_mask = ~attention_mask.bool()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        hidden_states, self_attention_p, cross_attention_p, key_value_states = self.decoder(
            self.embedding(target_ids.t()),
            encoder_output.transpose(0, 1),
            attention_mask,
            past_key_values
        )

        hidden_states = [e.transpose(0, 1) for e in hidden_states]
        last_layer = hidden_states[-1]
        hidden_states = [hidden_states[0]] + [
            hidden_states[i] - hidden_states[i - 1]
            for i in range(1, len(hidden_states))
        ]

        if not return_dict:
            return (
                last_layer,
                *([key_value_states] if use_cache else []),
                *([hidden_states] if output_hidden_states else []),
                *([self_attention_p] if output_attentions else []),
                *([cross_attention_p] if output_attentions else []),
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_layer,
            past_key_values=key_value_states if use_cache else None,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=self_attention_p if output_attentions else None,
            cross_attentions=cross_attention_p if output_attentions else None
        )


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        decoder_input_ids = self.set_decoder_special_tokens(decoder_input_ids)

        if encoder_outputs is None:
            encoder_outputs = self.get_encoder_output(
                input_ids, attention_mask, output_hidden_states, output_attentions, return_dict
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
    
        decoder_outputs = self.get_decoder_output(
            decoder_input_ids, encoder_outputs[0], attention_mask, past_key_values, use_cache, output_hidden_states, output_attentions, return_dict
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class NorT5ForConditionalGeneration(NorT5Model):

    def __init__(self, config):
        super().__init__(config, add_lm_layer=True)

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
    ):
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.get_encoder_output(
                input_ids, attention_mask, output_hidden_states, output_attentions, return_dict
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if labels is not None:
            labels = self.set_decoder_special_tokens(labels)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        elif decoder_input_ids is not None:
            decoder_input_ids = self.set_decoder_special_tokens(decoder_input_ids)

        decoder_outputs = self.get_decoder_output(
            decoder_input_ids, encoder_outputs[0], attention_mask, past_key_values, use_cache, output_hidden_states, output_attentions, return_dict
        )
        lm_logits = self.classifier(decoder_outputs[0])

        loss = None
        if labels is not None:
            labels.masked_fill_(labels == self.pad_token_id, -100)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.flatten(0, 1), labels.flatten())

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            print("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                layer_past_state = layer_past_state.index_select(0, beam_idx.to(layer_past_state.device))
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state,)

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class NorT5Encoder(NorT5Model):
    def __init__(self, config):
        super().__init__(config, add_lm_layer=False, add_decoder=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.get_encoder_output(
            input_ids, attention_mask, output_hidden_states, output_attentions, return_dict=return_dict
        )
