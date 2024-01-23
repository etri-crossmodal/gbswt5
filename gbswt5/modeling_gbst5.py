"""
    hf transformers-compatible GBST + T5 Model implementation.

    several methods are copying from huggingface/transformers/models/t5/modeling_t5.py
    as Implementation Standards for compatibility. (version 4.28.1)

    hf transformers' modeling_t5.py file is distributed under Apache 2.0 License.

    Copyright (C) 2023, ETRI LIRS, Jong-hun Shin.
"""
import copy
import sys

from typing import Optional, Union, Tuple

import torch

from torch import nn
from transformers import add_start_docstrings
from transformers.utils import logging
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5LayerNorm, T5Block, T5Stack,
    T5Model, T5PreTrainedModel, T5ForConditionalGeneration, T5EncoderModel,
    T5DenseActDense, T5DenseGatedActDense, T5Attention,
    T5_START_DOCSTRING
)

from .configuration_gbst5 import GBSWT5Config
from .gbst import GBSWT


logger = logging.get_logger(__name__)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class GBSWT5PreTrainedModel(T5PreTrainedModel):
    config_class = GBSWT5Config
    base_model_prefix = "GBSWT5"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights. 대부분은 T5PreTrainedModel을 따른다. """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            ( GBSWT5Model, GBSWT5ForConditionalGeneration, GBSWT5EncoderModel,),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        elif isinstance(module, GBSWT):
            module._init_weights(factor)


class GBSWT5Stack(GBSWT5PreTrainedModel):
    """ implement GBST-enabled T5Model, based on HF Transformers's T5Stack. """
    def __init__(self, config: GBSWT5Config, embed_tokens :nn.Embedding=None):
        # 초기화는 이전의 것을 따른다. 상속이 좀 애매해서, 사실 별도로 정의해야 하나 싶기도 하다.
        super().__init__(config)

        # override embed_tokens, apply GBWST
        self.embed_tokens = GBSWT(embed_tokens=embed_tokens,
                                  max_block_size=config.max_subword_block_size,
                                  blocks=config.subword_blocks,
                                  downsample_factor=config.downsample_factor,
                                  score_consensus_attn=config.score_consensus_attn,
                                  use_bn=config.gbst_batchnorm,)
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing, same as T5 Stack.
        self.post_init()
        # for Model Parallel
        self.model_parallel = False
        self.device_map = False
        self.gradient_checkpointing = False
        self.downsample_factor = config.downsample_factor

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        """ GBST 파트를 제외하면, T5Stack.forward() 구현을 그대로 복제하였다. """
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            #print(f"old: {input_shape}")
            inputs_embeds, attention_mask = self.embed_tokens(input_ids, attention_mask)
            # for downsample_factor > 1
            input_shape = inputs_embeds.size()[:-1]
            #print(f"new: {input_shape}")

        batch_size, seq_length = input_shape
        #print(f"bs: {batch_size}, sl: {seq_length}")

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        #print(f"mask_seq_length: {mask_seq_length}")

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # to return downsampled attention_mask,
        # hiding payload into last_hidden_states.
        # but you can get it from self.embed_tokens.get_resized_mask(attn_mask)
        setattr(hidden_states, 'attention_mask', attention_mask)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    def get_input_embeddings(self):
        return self.embed_tokens.embeds

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens.embeds = new_embeddings


GBSWT5Stack.parallelize = T5Stack.parallelize
GBSWT5Stack.deparallelize = T5Stack.deparallelize


class GBSWT5Model(GBSWT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.embeds.weight", "decoder_embed_tokens.embeds.weight"]

    def __init__(self, config: GBSWT5Config):
        """ override T5Model """
        # override some default missing parameters for pretrained ByT5 models (e.g. google/byt5-small)
        if not hasattr(config, 'max_subword_block_size'):
            config.max_subword_block_size = None
        if not hasattr(config, 'subword_blocks'):
            config.subword_blocks = ((1, 0), (2, 0), (3, 0), (6, 0), (9, 0),)
        if not hasattr(config, 'downsample_factor'):
            config.downsample_factor = 1
        if not hasattr(config, 'score_consensus_attn'):
            config.score_consensus_attn = True

        super().__init__(config)

        # naive T5와 같이 embedding은 공유함
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_cfg = copy.deepcopy(config)
        encoder_cfg.is_decoder = False
        encoder_cfg.use_cache = False
        encoder_cfg.is_encoder_decoder = False
        self.encoder = GBSWT5Stack(encoder_cfg, self.shared)

        # Embedding base를 공유하기는 하지만, decoder에는 GBSWT를
        # 적용하지 않아야 한다.
        decoder_cfg = copy.deepcopy(config)
        decoder_cfg.is_decoder = True
        decoder_cfg.is_encoder_decoder = False
        decoder_cfg.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_cfg, self.shared)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                decoder_inputs_embeds: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        """
        중요한 것은, downsampling이 된 경우 attention_mask가 변경되므로,
        이를 반영해주는 것이 필요하다. hf transformers 4.29.1에서 복제함
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # update attention mask
            if hasattr(encoder_outputs[0], 'attention_mask'):
                attention_mask = getattr(encoder_outputs[0], 'attention_mask')

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # resize attn_mask when it mismatched
        if attention_mask is not None and hidden_states.size(1) != attention_mask.size(1):
            attention_mask = self.encoder.embed_tokens.get_resized_mask(attention_mask)

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

        # Decode
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


GBSWT5Model.parallelize = T5Model.parallelize
GBSWT5Model.deparallelize = T5Model.deparallelize
GBSWT5Model.get_input_embeddings = T5Model.get_input_embeddings
GBSWT5Model.set_input_embeddings = T5Model.set_input_embeddings
GBSWT5Model.get_encoder = T5Model.get_encoder
GBSWT5Model._prune_heads = T5Model._prune_heads


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class GBSWT5ForConditionalGeneration(GBSWT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.embeds.weight",
                          "decoder_embed_tokens.embeds.weight",
                          "lm_head.weight"]

    def __init__(self, config: GBSWT5Config):
        # override some default missing parameters for pretrained ByT5 models (e.g. google/byt5-small)
        if not hasattr(config, 'max_subword_block_size'):
            config.max_subword_block_size = None
        if not hasattr(config, 'subword_blocks'):
            config.subword_blocks = ((1, 0), (2, 0), (3, 0), (6, 0), (9, 0),)
        if not hasattr(config, 'downsample_factor'):
            config.downsample_factor = 1
        if not hasattr(config, 'score_consensus_attn'):
            config.score_consensus_attn = True

        # Grandparent의 init를 그대로 상속, 나머지는 T5ForConditionalGeneration을 따름
        super().__init__(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_cfg = copy.deepcopy(config)
        encoder_cfg.is_decoder = False
        encoder_cfg.use_cache = False
        encoder_cfg.is_encoder_decoder = False
        self.encoder = GBSWT5Stack(encoder_cfg, self.shared)

        # Embedding base를 공유하기는 하지만, decoder에는 GBSWT를
        # 적용하지 않아야 한다.
        decoder_cfg = copy.deepcopy(config)
        decoder_cfg.is_decoder = True
        decoder_cfg.is_encoder_decoder = False
        decoder_cfg.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_cfg, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(self,
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
                ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        중요한 것은 encoder outputs에서 수정된 attention_mask를 다시 반영해야 하는 것임
        downsampling이 들어간 경우, attention_mask가 변경되기 때문.
        """
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

            # update attention mask
            if hasattr(encoder_outputs[0], 'attention_mask'):
                attention_mask = getattr(encoder_outputs[0], 'attention_mask')
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        # resize attn_mask when it mismatched
        if attention_mask is not None and hidden_states.size(1) != attention_mask.size(1):
            attention_mask = self.encoder.embed_tokens.get_resized_mask(attention_mask)

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

        # Decode
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

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # add z_loss for computational stability in bf16 amp.
            # see https://github.com/huggingface/transformers/pull/10956#issuecomment-820712267
            if self.config.z_loss != 0.0:
                log_z = lm_logits.view(-1).logsumexp(-1)
                loss += self.config.z_loss * log_z.square()

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


GBSWT5ForConditionalGeneration.parallelize = T5ForConditionalGeneration.parallelize
GBSWT5ForConditionalGeneration.deparallelize = T5ForConditionalGeneration.deparallelize
GBSWT5ForConditionalGeneration.get_input_embeddings = T5ForConditionalGeneration.get_input_embeddings
GBSWT5ForConditionalGeneration.set_input_embeddings = T5ForConditionalGeneration.set_input_embeddings
GBSWT5ForConditionalGeneration.get_output_embeddings = T5ForConditionalGeneration.get_output_embeddings
GBSWT5ForConditionalGeneration.set_output_embeddings = T5ForConditionalGeneration.set_output_embeddings
GBSWT5ForConditionalGeneration.get_encoder = T5ForConditionalGeneration.get_encoder
GBSWT5ForConditionalGeneration.prepare_inputs_for_generation = T5ForConditionalGeneration.prepare_inputs_for_generation
GBSWT5ForConditionalGeneration.prepare_decoder_input_ids_from_labels = T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels
GBSWT5ForConditionalGeneration._reorder_cache = T5ForConditionalGeneration._reorder_cache
GBSWT5ForConditionalGeneration._prune_heads = T5Model._prune_heads


class GBSWT5EncoderModel(T5PreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.embeds.weight"]

    def __init__(self, config: GBSWT5Config):
        # override some default missing parameters for pretrained ByT5 models (e.g. google/byt5-small)
        if not hasattr(config, 'max_subword_block_size'):
            config.max_subword_block_size = None
        if not hasattr(config, 'subword_blocks'):
            config.subword_blocks = ((1, 0), (2, 0), (3, 0), (6, 0), (9, 0),)
        if not hasattr(config, 'downsample_factor'):
            config.downsample_factor = 1
        if not hasattr(config, 'score_consensus_attn'):
            config.score_consensus_attn = True

        super().__init__(config)

        # naive T5와 같이 embedding은 공유함
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_cfg = copy.deepcopy(config)
        encoder_cfg.is_decoder = False
        encoder_cfg.use_cache = False
        encoder_cfg.is_encoder_decoder = False
        self.encoder = GBSWT5Stack(encoder_cfg, self.shared)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                return_resized_attention_mask: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        downsampled 된 attention_mask를 함께 반환한다. 단, return_resized_attention_mask=True일 때만.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_resized_attention_mask:
            attention_mask = self.encoder.embed_tokens.get_resized_mask(attention_mask)
            return encoder_outputs, attention_mask

        return encoder_outputs


GBSWT5EncoderModel.parallelize = T5EncoderModel.parallelize
GBSWT5EncoderModel.deparallelize = T5EncoderModel.deparallelize
GBSWT5EncoderModel.get_input_embeddings = T5EncoderModel.get_input_embeddings
GBSWT5EncoderModel.set_input_embeddings = T5EncoderModel.set_input_embeddings
GBSWT5EncoderModel.get_encoder = T5EncoderModel.get_encoder
GBSWT5EncoderModel._prune_heads = T5EncoderModel._prune_heads
