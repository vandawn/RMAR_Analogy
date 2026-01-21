from typing import Any, Optional, Tuple
import math

import torch
from torch import nn, Tensor, device
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from copy import deepcopy
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput, 
    MaskedLMOutput,
    BaseModelOutputWithPooling,
)

# some function
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        head_mask = [None] * num_hidden_layers

        return head_mask


# models

class GatedFusion(nn.Module):
    """
    一个自定义的门控融合单元
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        # 门控层，学习一个0-1之间的权重
        self.linear_gate = nn.Linear(hidden_size * 2, hidden_size)
        # 特征变换层
        self.linear_transform = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, base_vec, struct_vec):
        # base_vec 是基础模态 (如文本/图像)，struct_vec 是要融入的结构模态
        gate = torch.sigmoid(self.linear_gate(torch.cat([base_vec, struct_vec], dim=-1)))

        transformed_features = torch.tanh(self.linear_transform(torch.cat([base_vec, struct_vec], dim=-1)))

        fused_vec = (1 - gate) * base_vec + gate * transformed_features
        fused_vec = self.dropout(fused_vec)
        fused_vec = self.layer_norm(fused_vec + base_vec) # 使用残差连接增加稳定性

        # 同时返回融合后的向量和门控权重
        return fused_vec, gate
    
    
class UnimoConfig(PretrainedConfig):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UnimoPreTrainedModel(PreTrainedModel):
    config_class = UnimoConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init_weights(self, module):
        pass


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

    def forward(self, pixel_values):
        # pixel_values: (bsz, 2, 3, 224, 224)
        batch_size = pixel_values.shape[0]
        patch_embeds = torch.cat([
                            self.patch_embedding(pixel_values[:, 0]).flatten(2).transpose(1, 2),
                            self.patch_embedding(pixel_values[:, 1]).flatten(2).transpose(1, 2)], 
                            dim=1
                        )   # bsz, 98, 768
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + torch.cat([self.position_embedding(self.position_ids), self.position_embedding(self.position_ids)[:, 1:]], dim=1)

        return embeddings


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )       
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads   # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.fusion = BertFusion(config)    # 
        
        # adaptive analogy mask
        self.adaptive_weight = nn.ParameterList([
            #    nn.Parameter(torch.FloatTensor(1).uniform_(1.0, 2.5)),  # example to query
            #    nn.Parameter(torch.FloatTensor(1).uniform_(1.0, 2.5))   # query to example
                nn.Parameter(torch.FloatTensor(1).uniform_(0.0, 0.5)),  # example to query
                nn.Parameter(torch.FloatTensor(1).uniform_(0.5, 0.5))   # query to example
        ])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        sep_idx=None
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        qks = (key_layer, value_layer) if output_qks else None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if sep_idx is not None:
            for i, idx in enumerate(sep_idx):
                # example to answer
                # attention_scores[i, :, :idx[2], idx[2]:] = torch.sigmoid(self.adaptive_weight[0]) * attention_scores[i, :, :idx[2], idx[2]:].clone()
                attention_scores[i, :, :idx[2], idx[2]:] = torch.clamp(self.adaptive_weight[0], 0, 0.5) * attention_scores[i, :, :idx[2], idx[2]:].clone()
                # answer to example
                # attention_scores[i, :, idx[2]:, idx[2]:] = torch.sigmoid(self.adaptive_weight[1]) * attention_scores[i, :, idx[2]:, idx[2]:].clone()
                attention_scores[i, :, idx[2]:, idx[2]:] = torch.clamp(self.adaptive_weight[1], 0.5, 1) * attention_scores[i, :, idx[2]:, idx[2]:].clone()
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            '''add adaptive analogy mask, attention_scores ~ (bsz, 12, seq_len, seq_len), attention_mask ~ (bsz, 1, seq_len, seq_len)'''
            
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)    # bsz, 128, 768
        
        fusion_output = self.fusion(context_layer, visual_hidden_state) if visual_hidden_state is not None else None # add

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, fusion_output, qks


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.fusion_function = config.fusion_function
        self.fusion_function = 'softmax'

    def forward(
        self,
        hidden_states,
        visual_hidden_state=None,
    ):
        fusion_scores = torch.matmul(hidden_states, visual_hidden_state.transpose(-1, -2))  # bsz, 128, 49
        # if attention_mask is not None:
        #     # attention_mask: bsz, 1, 1, 128; fusion_scores: bsz, 128, 49
        #     fusion_scores = fusion_scores + attention_mask.squeeze(1).transpose(1, 2)
        if self.fusion_function == 'softmax':
            fusion_probs = nn.Softmax(dim=-1)(fusion_scores)
            fusion_output = torch.matmul(fusion_probs, visual_hidden_state)
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(dim=-1)
        return fusion_output
    

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        sep_idx=None,
    ):
        self_outputs, fusion_output, qks = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            visual_hidden_state,
            output_qks,
            sep_idx
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, fusion_output, qks


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fusion_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, fusion_output=None):
        hidden_states = self.dense(hidden_states)
        if fusion_output is not None:
            fusion_states = self.fusion_dense(fusion_output)
            hidden_states = hidden_states + fusion_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
    
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        sep_idx=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs, fusion_output, qks = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            visual_hidden_state=visual_hidden_state,
            output_qks=output_qks,
            sep_idx=sep_idx,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, fusion_output
        )
        outputs = (layer_output,) + outputs
        if output_qks: 
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output, fusion_output):
        intermediate_output = self.intermediate(attention_output, fusion_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class UnimoEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])
    
    def forward(
        self,
        vision_embeds=None,
        text_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sep_idx=None,
    ):
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None
        
        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )
            
            # vision
            # TODO: 9-12 layers past text as pkv to vision
            past_key_values = text_layer_output[-1] if idx >= 8 else None
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                    vision_hidden_states,
                    output_attentions=output_attentions,
                    past_key_values=past_key_values,
            )
            vision_hidden_states = vision_layer_output[0]

            # text
            # TODO: 9-12 layers past vison qks to text
            last_hidden_state = vision_hidden_states if idx >= 8 else None
            output_qks = True if idx >= 7 else None
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(
                    text_hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    visual_hidden_state=last_hidden_state,
                    output_attentions=output_attentions,
                    output_qks=output_qks,
                    sep_idx=sep_idx,
            )
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1], )
                all_text_attentions = all_text_attentions + (text_layer_output[1], )
        
        if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )
        
        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)
        return BaseModelOutput(
            last_hidden_state=text_hidden_states, hidden_states=all_text_hidden_states, attentions=all_text_attentions
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# class UnimoModel(nn.Module):
#     def __init__(self, vision_config, text_config, args, add_pooling_layer=True, structural_embeddings=None):
#         super(UnimoModel, self).__init__()
#         self.args = args
#         # vision model
#         self.vision_config = vision_config
#         self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
#         self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
#         self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

#         # text model
#         self.text_config = text_config
#         self.text_embeddings = BertEmbeddings(text_config)
#         self.text_pooler = BertPooler(text_config) if add_pooling_layer else None
        
#         # structure model
#         self.struct_hidden_size = 0
#         if structural_embeddings is not None:
#             self.structural_embedding = nn.Embedding.from_pretrained(
#                 embeddings=structural_embeddings,
#                 freeze=False  # 设为False可以微调，True则冻结
#             )
#             self.struct_hidden_size = self.structural_embedding.weight.size(1)
            
#             # +++ 根据 fusion_strategy 创建不同的融合层 +++
            
#             self.fusion_strategy = self.args.fusion_strategy
#             if self.fusion_strategy == 'concat':
#                 # 创建两个独立的融合层，一个用于文本，一个用于图像，这样更灵活
#                 fusion_dim = text_config.hidden_size + self.struct_hidden_size

#                 # 文本融合层
#                 self.text_fusion_layer = nn.Sequential(
#                     nn.Linear(fusion_dim, text_config.hidden_size),
#                     nn.ReLU(),
#                     nn.Dropout(0.1)
#                 )
#                 # 图像融合层
#                 self.image_fusion_layer = nn.Sequential(
#                     nn.Linear(fusion_dim, vision_config.hidden_size),
#                     nn.ReLU(),
#                     nn.Dropout(0.1)
#                 )
#             elif self.fusion_strategy == 'gate':
#                 self.text_fusion_layer = GatedFusion(text_config.hidden_size)
#                 self.image_fusion_layer = GatedFusion(vision_config.hidden_size)

#         # all
#         self.encoder = UnimoEncoder(vision_config, text_config)

#         self.device = vision_config.device

    
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         sep_idx=None,
#         pixel_values=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         # 从数据加载器传来的新参数
#         struct_head_id=None,
#         struct_rel_id=None,
#         struct_tail_id=None,
#         head_token_idx=None,
#         tail_token_idx=None,
#         struct_a_head_id=None,
#         a_head_token_idx=None
#     ):
#         # 1. 获取原始的文本和图像Embedding
#         text_embedding_output = self.text_embeddings(
#             input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
#         )
#         vision_embedding_output = self.vision_embeddings(pixel_values)
#         vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

#         # 创建最终要传入encoder的副本
#         text_embedding_output_fused = text_embedding_output.clone()
#         vision_embedding_output_fused = vision_embedding_output.clone()
        
#         batch_size = input_ids.shape[0]
#         gates = {}

#         # 2. 如果存在结构化信息，则进行融合
#         if hasattr(self, 'structural_embedding') and self.structural_embedding is not None:
            
#             # --- 新增：提前取出所有需要融合的原始向量 ---
#             # 关键：所有“读”操作，都从原始的、未被修改的张量中进行
            
#             # 文本部分
#             head_text_vecs, tail_text_vecs, a_head_text_vecs = None, None, None
#             valid_head_mask = (head_token_idx != -1) if head_token_idx is not None else None
#             if valid_head_mask is not None and valid_head_mask.any():
#                 head_text_vecs = text_embedding_output[valid_head_mask, head_token_idx[valid_head_mask]]

#             valid_tail_mask = (tail_token_idx != -1) if tail_token_idx is not None else None
#             if valid_tail_mask is not None and valid_tail_mask.any():
#                 tail_text_vecs = text_embedding_output[valid_tail_mask, tail_token_idx[valid_tail_mask]]
            
#             valid_a_head_mask = (a_head_token_idx != -1) if a_head_token_idx is not None else None
#             if valid_a_head_mask is not None and valid_a_head_mask.any():
#                 a_head_text_vecs = text_embedding_output[valid_a_head_mask, a_head_token_idx[valid_a_head_mask]]

#             # 图像部分
#             head_image_cls_vec = vision_embedding_output[:, 0, :] if struct_head_id is not None else None
#             num_vision_positions = self.vision_embeddings.num_positions
#             tail_image_cls_idx = num_vision_positions
#             tail_image_cls_vec = vision_embedding_output[:, tail_image_cls_idx, :] if struct_tail_id is not None else None

#             # --- 新增：计算出所有融合后的新向量 ---
            
#             # 文本部分
#             fused_head_text, fused_tail_text, fused_a_head_text = None, None, None
#             if head_text_vecs is not None:
#                 head_struct_vecs = self.structural_embedding(struct_head_id[valid_head_mask])
#                 if self.fusion_strategy == 'concat':
#                     fused_head_text = self.text_fusion_layer(torch.cat([head_text_vecs, head_struct_vecs], dim=-1))
#                 elif self.fusion_strategy == 'gate':
#                     fused_head_text, gate = self.text_fusion_layer(head_text_vecs, head_struct_vecs)
#                     gates['text_head_gate'] = gate
            
#             if tail_text_vecs is not None:
#                 tail_struct_vecs = self.structural_embedding(struct_tail_id[valid_tail_mask])
#                 if self.fusion_strategy == 'concat':
#                     fused_tail_text = self.text_fusion_layer(torch.cat([tail_text_vecs, tail_struct_vecs], dim=-1))
#                 elif self.fusion_strategy == 'gate':
#                     fused_tail_text, gate = self.text_fusion_layer(tail_text_vecs, tail_struct_vecs)
#                     gates['text_tail_gate'] = gate
            
#             if a_head_text_vecs is not None:
#                 a_head_struct_vecs = self.structural_embedding(struct_a_head_id[valid_a_head_mask])
#                 if self.fusion_strategy == 'concat':
#                     fused_a_head_text = self.text_fusion_layer(torch.cat([a_head_text_vecs, a_head_struct_vecs], dim=-1))
#                 elif self.fusion_strategy == 'gate':
#                     fused_a_head_text, gate = self.text_fusion_layer(a_head_text_vecs, a_head_struct_vecs)
#                     gates['text_a_head_gate'] = gate

#             # 图像部分
#             fused_head_image, fused_tail_image = None, None
#             if head_image_cls_vec is not None:
#                 head_struct_vecs = self.structural_embedding(struct_head_id)
#                 if self.fusion_strategy == 'concat':
#                     fused_head_image = self.image_fusion_layer(torch.cat([head_image_cls_vec, head_struct_vecs], dim=-1))
#                 elif self.fusion_strategy == 'gate':
#                     fused_head_image, gate = self.image_fusion_layer(head_image_cls_vec, head_struct_vecs)
#                     gates['image_head_gate'] = gate

#             if tail_image_cls_vec is not None:
#                 tail_struct_vecs = self.structural_embedding(struct_tail_id)
#                 if self.fusion_strategy == 'concat':
#                     fused_tail_image = self.image_fusion_layer(torch.cat([tail_image_cls_vec, tail_struct_vecs], dim=-1))
#                 elif self.fusion_strategy == 'gate':
#                     fused_tail_image, gate = self.image_fusion_layer(tail_image_cls_vec, tail_struct_vecs)
#                     gates['image_tail_gate'] = gate
            
#             # --- 新增：现在，统一将所有新向量写入副本 ---
#             # 关键：所有的“写”操作，都发生在副本上
#             if fused_head_text is not None:
#                 text_embedding_output_fused[valid_head_mask, head_token_idx[valid_head_mask]] = fused_head_text
#             if fused_tail_text is not None:
#                 text_embedding_output_fused[valid_tail_mask, tail_token_idx[valid_tail_mask]] = fused_tail_text
#             if fused_a_head_text is not None:
#                 text_embedding_output_fused[valid_a_head_mask, a_head_token_idx[valid_a_head_mask]] = fused_a_head_text
            
#             if fused_head_image is not None:
#                 vision_embedding_output_fused[:, 0, :] = fused_head_image
#             if fused_tail_image is not None:
#                 vision_embedding_output_fused[:, tail_image_cls_idx, :] = fused_tail_image

#         # 3. 将增强后的Embedding副本送入主编码器
#         input_shape = input_ids.size()
#         extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device=input_ids.device)
#         head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)

#         encoder_outputs = self.encoder(
#             vision_embeds=vision_embedding_output_fused,
#             text_embeds=text_embedding_output_fused,
#             attention_mask=extended_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             sep_idx=sep_idx,
#         )
        
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

#         if not return_dict:
#             return (sequence_output, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPooling(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         ), gates


class UnimoModel(nn.Module):
    def __init__(self, vision_config, text_config, args, add_pooling_layer=True, structural_embeddings=None):
        super(UnimoModel, self).__init__()
        self.args = args
        self.use_structure = bool(getattr(args, "use_structure", 1))

        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # structure model
        self.struct_hidden_size = 0
        if (structural_embeddings is not None) and self.use_structure:
            self.structural_embedding = nn.Embedding.from_pretrained(
                embeddings=structural_embeddings,
                freeze=False
            )
            self.struct_hidden_size = self.structural_embedding.weight.size(1)

            # fusion layers by strategy
            self.fusion_strategy = self.args.fusion_strategy
            if self.fusion_strategy == 'concat':
                fusion_dim_t = text_config.hidden_size + self.struct_hidden_size
                fusion_dim_v = vision_config.hidden_size + self.struct_hidden_size
                self.text_fusion_layer = nn.Sequential(
                    nn.Linear(fusion_dim_t, text_config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                self.image_fusion_layer = nn.Sequential(
                    nn.Linear(fusion_dim_v, vision_config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            elif self.fusion_strategy == 'gate':
                self.text_fusion_layer = GatedFusion(text_config.hidden_size)
                self.image_fusion_layer = GatedFusion(vision_config.hidden_size)

        # encoder
        self.encoder = UnimoEncoder(vision_config, text_config)
        self.device = vision_config.device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        sep_idx=None,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # 从数据加载器传来的新参数
        struct_head_id=None,
        struct_rel_id=None,
        struct_tail_id=None,
        head_token_idx=None,
        tail_token_idx=None,
        struct_a_head_id=None,
        a_head_token_idx=None
    ):
        # 1) 原始文本/图像 Embedding
        text_embedding_output = self.text_embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        vision_embedding_output = self.vision_embeddings(pixel_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # 创建副本（所有“写”发生在副本上）
        text_embedding_output_fused = text_embedding_output.clone()
        vision_embedding_output_fused = vision_embedding_output.clone()

        gates = {}

        # 2) 结构融合（可关）
        if self.use_structure and hasattr(self, 'structural_embedding') and self.structural_embedding is not None:

            # 准备文本位置向量（只从原始张量读取）
            head_text_vecs, tail_text_vecs, a_head_text_vecs = None, None, None
            valid_head_mask = (head_token_idx != -1) if head_token_idx is not None else None
            if valid_head_mask is not None and valid_head_mask.any():
                head_text_vecs = text_embedding_output[valid_head_mask, head_token_idx[valid_head_mask]]

            valid_tail_mask = (tail_token_idx != -1) if tail_token_idx is not None else None
            if valid_tail_mask is not None and valid_tail_mask.any():
                tail_text_vecs = text_embedding_output[valid_tail_mask, tail_token_idx[valid_tail_mask]]

            valid_a_head_mask = (a_head_token_idx != -1) if a_head_token_idx is not None else None
            if valid_a_head_mask is not None and valid_a_head_mask.any():
                a_head_text_vecs = text_embedding_output[valid_a_head_mask, a_head_token_idx[valid_a_head_mask]]

            # 图像 CLS
            head_image_cls_vec = vision_embedding_output[:, 0, :] if struct_head_id is not None else None
            num_vision_positions = self.vision_embeddings.num_positions
            tail_image_cls_idx = num_vision_positions
            tail_image_cls_vec = vision_embedding_output[:, tail_image_cls_idx, :] if struct_tail_id is not None else None

            # 计算融合后的新向量
            fused_head_text = fused_tail_text = fused_a_head_text = None
            if head_text_vecs is not None:
                head_struct_vecs = self.structural_embedding(struct_head_id[valid_head_mask])
                if self.fusion_strategy == 'concat':
                    fused_head_text = self.text_fusion_layer(torch.cat([head_text_vecs, head_struct_vecs], dim=-1))
                else:
                    fused_head_text, gate = self.text_fusion_layer(head_text_vecs, head_struct_vecs)
                    gates['text_head_gate'] = gate

            if tail_text_vecs is not None:
                tail_struct_vecs = self.structural_embedding(struct_tail_id[valid_tail_mask])
                if self.fusion_strategy == 'concat':
                    fused_tail_text = self.text_fusion_layer(torch.cat([tail_text_vecs, tail_struct_vecs], dim=-1))
                else:
                    fused_tail_text, gate = self.text_fusion_layer(tail_text_vecs, tail_struct_vecs)
                    gates['text_tail_gate'] = gate

            if a_head_text_vecs is not None:
                a_head_struct_vecs = self.structural_embedding(struct_a_head_id[valid_a_head_mask])
                if self.fusion_strategy == 'concat':
                    fused_a_head_text = self.text_fusion_layer(torch.cat([a_head_text_vecs, a_head_struct_vecs], dim=-1))
                else:
                    fused_a_head_text, gate = self.text_fusion_layer(a_head_text_vecs, a_head_struct_vecs)
                    gates['text_a_head_gate'] = gate

            fused_head_image = fused_tail_image = None
            if head_image_cls_vec is not None:
                head_struct_vecs = self.structural_embedding(struct_head_id)
                if self.fusion_strategy == 'concat':
                    fused_head_image = self.image_fusion_layer(torch.cat([head_image_cls_vec, head_struct_vecs], dim=-1))
                else:
                    fused_head_image, gate = self.image_fusion_layer(head_image_cls_vec, head_struct_vecs)
                    gates['image_head_gate'] = gate

            if tail_image_cls_vec is not None:
                tail_struct_vecs = self.structural_embedding(struct_tail_id)
                if self.fusion_strategy == 'concat':
                    fused_tail_image = self.image_fusion_layer(torch.cat([tail_image_cls_vec, tail_struct_vecs], dim=-1))
                else:
                    fused_tail_image, gate = self.image_fusion_layer(tail_image_cls_vec, tail_struct_vecs)
                    gates['image_tail_gate'] = gate

            # 写回副本
            if fused_head_text is not None:
                text_embedding_output_fused[valid_head_mask, head_token_idx[valid_head_mask]] = fused_head_text
            if fused_tail_text is not None:
                text_embedding_output_fused[valid_tail_mask, tail_token_idx[valid_tail_mask]] = fused_tail_text
            if fused_a_head_text is not None:
                text_embedding_output_fused[valid_a_head_mask, a_head_token_idx[valid_a_head_mask]] = fused_a_head_text
            if fused_head_image is not None:
                vision_embedding_output_fused[:, 0, :] = fused_head_image
            if fused_tail_image is not None:
                vision_embedding_output_fused[:, tail_image_cls_idx, :] = fused_tail_image

        # 3) 编码器
        input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device=input_ids.device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)

        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output_fused,
            text_embeds=text_embedding_output_fused,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            sep_idx=sep_idx,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), gates


    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

        
class UnimoForMaskedLM(nn.Module):
    def __init__(self, vision_config, text_config, args, structural_embeddings=None):
        super().__init__()
        self.args = args
        self.unimo = UnimoModel(vision_config, text_config, args, structural_embeddings=structural_embeddings)
        self.cls = UnimoOnlyMLMHead(text_config)
        self.config = text_config
        
        
        # +++ 新增组件 +++
        # 1. 新增关系编码器和注意力融合模块
        #    假设关系嵌入维度 D_r 和 [R] 维度 D_R_fine 都是 hidden_size
        hidden_size = text_config.hidden_size
        self.relation_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.neighborhood_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=text_config.num_attention_heads, batch_first=True)

        # 2. (进阶) 新增动量编码器
        #    这里我们复制主模型unimo和关系编码器
        #    deepcopy确保了参数是独立的
        
        self.unimo_m = deepcopy(self.unimo)
        self.relation_encoder_m = deepcopy(self.relation_encoder)
        # 冻结动量编码器的参数，因为它们不通过梯度下降更新
        for param in self.unimo_m.parameters():
            param.requires_grad = False
        for param in self.relation_encoder_m.parameters():
            param.requires_grad = False
            
        # 3. 新增记忆库 (Memory Bank)
        self.memory_k = args.memory_size  # 例如 4096, 在run脚本中配置
        self.momentum_m = args.momentum_m # 例如 0.999, 在run脚本中配置
        
        # 使用 register_buffer 将它们注册为模型的状态，但不作为可训练参数
        # 它们会随着模型一起被移动到GPU上
        self.register_buffer("memory_r_emb", torch.randn(self.memory_k, hidden_size))
        self.register_buffer("memory_R_fine", torch.randn(self.memory_k, hidden_size))
        # 重要：记忆库也需要存储关系标签，用于调试打印
        self.register_buffer("memory_rel_label", torch.full((self.memory_k,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.tie_weights()

    @torch.no_grad()
    def _update_momentum_encoder(self):
        """动量更新孪生编码器"""
        for param_q, param_m in zip(self.unimo.parameters(), self.unimo_m.parameters()):
            param_m.data = param_m.data * self.momentum_m + param_q.data * (1. - self.momentum_m)
        for param_q, param_m in zip(self.relation_encoder.parameters(), self.relation_encoder_m.parameters()):
            param_m.data = param_m.data * self.momentum_m + param_q.data * (1. - self.momentum_m)
    
    # # 简单队列，先进先出       
    @torch.no_grad()
    def _update_memory_bank(self, r_embs, R_fines, rel_labels):
        #import pdb; pdb.set_trace()
        """更新记忆库"""
        batch_size = r_embs.shape[0]
        ptr = int(self.queue_ptr)
        
        # 确保不会超出范围
        #assert self.memory_k % batch_size == 0
        
        # 计算新数据将要填充的位置
        end_ptr = ptr + batch_size

        if end_ptr <= self.memory_k:
            # Case 1: 新数据不会超出队列末尾，直接整块替换
            self.memory_r_emb[ptr:end_ptr, :] = r_embs
            self.memory_R_fine[ptr:end_ptr, :] = R_fines
            self.memory_rel_label[ptr:end_ptr] = rel_labels
        else:
            # Case 2: 新数据会“绕圈”，需要分两部分填充
            # 第一部分：填满队列末尾
            num_to_end = self.memory_k - ptr
            self.memory_r_emb[ptr:, :] = r_embs[:num_to_end]
            self.memory_R_fine[ptr:, :] = R_fines[:num_to_end]
            self.memory_rel_label[ptr:] = rel_labels[:num_to_end]

            # 第二部分：从队列开头继续填充剩余部分
            num_from_start = end_ptr - self.memory_k
            self.memory_r_emb[:num_from_start, :] = r_embs[num_to_end:]
            self.memory_R_fine[:num_from_start, :] = R_fines[num_to_end:]
            self.memory_rel_label[:num_from_start] = rel_labels[num_to_end:]

        # 更新指针
        ptr = end_ptr % self.memory_k
        self.queue_ptr[0] = ptr
        
         # +++ 新增组件 +++
    
    # # 相似度阈值替换
    # @torch.no_grad()
    # def _update_memory_bank(self, r_embs, R_fines, rel_labels):
    #     """
    #     更新记忆库，采用相似度阈值策略增强多样性。
    #     """
    #     # 确保模型可以访问到args中定义的超参数
    #     # 您的代码已在__init__中保存了args (self.args = args)，所以这里可以直接使用
    #     diversity_threshold = self.args.diversity_threshold
    #     batch_size = r_embs.shape[0]

    #     # 从批处理更新改为逐个样本更新，以实现条件逻辑
    #     for i in range(batch_size):
    #         r_emb_new = r_embs[i].unsqueeze(0) # 增加一个维度以进行批量计算
            
    #         # 1. 计算新样本与记忆库中所有样本的余弦相似度
    #         #    我们只关心r_emb的相似度
    #         #    需要处理记忆库中可能存在的未初始化向量（标签为-1）
    #         valid_memory_mask = self.memory_rel_label != -1
    #         if not valid_memory_mask.any(): # 如果记忆库为空，直接添加
    #             max_sim = -1.0 
    #         else:
    #             sim_scores = torch.nn.functional.cosine_similarity(r_emb_new, self.memory_r_emb[valid_memory_mask])
    #             max_sim, most_similar_idx_local = torch.max(sim_scores, dim=0)
    #             # 将局部索引转换为全局索引
    #             most_similar_idx_global = torch.where(valid_memory_mask)[0][most_similar_idx_local]


    #         # 2. 根据相似度阈值进行决策
    #         if max_sim > diversity_threshold:
    #             # 2.1 高相似度：新样本是冗余的，用它替换掉最相似的那个旧样本
    #             # 这种情况下，FIFO队列的指针不移动，因为我们没有替换“最老”的元素
    #             idx_to_replace = most_similar_idx_global
                
    #             self.memory_r_emb[idx_to_replace, :] = r_embs[i]
    #             self.memory_R_fine[idx_to_replace, :] = R_fines[i]
    #             self.memory_rel_label[idx_to_replace] = rel_labels[i]

    #         else:
    #             # 2.2 低相似度：新样本是“新颖”的，采用FIFO逻辑
    #             # 获取当前最老元素的位置指针
    #             ptr = int(self.queue_ptr)
    #             idx_to_replace = ptr
                
    #             self.memory_r_emb[idx_to_replace, :] = r_embs[i]
    #             self.memory_R_fine[idx_to_replace, :] = R_fines[i]
    #             self.memory_rel_label[idx_to_replace] = rel_labels[i]
                
    #             # 更新指针，使其指向下一个最老的位置
    #             self.queue_ptr[0] = (ptr + 1) % self.memory_k
         
    #直接替换最相似的实体          
    # @torch.no_grad()
    # def _update_memory_bank(self, r_embs, R_fines, rel_labels):
    #     """
    #     更新记忆库，采用混合策略（新样本替换最相似的旧样本）以增强多样性。
    #     注意：此策略不再使用FIFO队列指针 queue_ptr。
    #     """
    #     batch_size = r_embs.shape[0]

    #     # 逐个处理批次中的新样本
    #     for i in range(batch_size):
    #         r_emb_new = r_embs[i].unsqueeze(0) # 增加一个维度以进行批量计算
            
    #         # 1. 检查记忆库中是否有未初始化的槽位 (rel_label == -1)
    #         #    优先填充这些空槽位，而不是替换已有样本
    #         uninitialized_slots = torch.where(self.memory_rel_label == -1)[0]

    #         if len(uninitialized_slots) > 0:
    #             # 1.1 如果有空槽位，则将新样本放入第一个空槽位
    #             idx_to_replace = uninitialized_slots[0]
    #         else:
    #             # 1.2 如果记忆库已满，则计算相似度，找到最相似的旧样本
    #             sim_scores = torch.nn.functional.cosine_similarity(r_emb_new, self.memory_r_emb)
    #             # 使用 argmax 找到相似度最高的旧样本的索引
    #             idx_to_replace = torch.argmax(sim_scores)

    #         # 2. 在计算出的位置用新样本进行替换
    #         self.memory_r_emb[idx_to_replace, :] = r_embs[i]
    #         self.memory_R_fine[idx_to_replace, :] = R_fines[i]
    #         self.memory_rel_label[idx_to_replace] = rel_labels[i]
            
    #     # 在此策略下，self.queue_ptr 不再被使用或更新。
        
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        sep_idx=None,
        
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        struct_head_id=None,
        struct_rel_id=None,
        struct_tail_id=None,
        head_token_idx=None,
        tail_token_idx=None,
        # v-- 在这里被定义为函数参数 --v
        struct_a_head_id=None,
        a_head_token_idx=None,
    ):
        outputs, gates = self.unimo(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            sep_idx=sep_idx,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # +++ 将参数传递给 self.unimo +++
            struct_head_id=struct_head_id,
            struct_tail_id=struct_tail_id,
            head_token_idx=head_token_idx,
            tail_token_idx=tail_token_idx,
            struct_a_head_id=struct_a_head_id,
            a_head_token_idx=a_head_token_idx
        )

        sequence_output = outputs[0]
        prediction_scores, trans_hidden_states = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), trans_hidden_states, gates

    def get_input_embeddings(self):
        return self.unimo.text_embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        self._tie_or_clone_weights(output_embeddings, self.unimo.get_input_embeddings())

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(self, new_num_tokens):
        self.unimo.resize_token_embeddings(new_num_tokens)
        # +++ 新增：同步调整动量模型的嵌入大小 +++
        self.unimo_m.resize_token_embeddings(new_num_tokens)
        self.tie_weights()

class UnimoOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = UnimoLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores, trans_hidden_states = self.predictions(sequence_output)
        return prediction_scores, trans_hidden_states


class UnimoLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        trans_hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(trans_hidden_states)
        return hidden_states, trans_hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states