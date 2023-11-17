"""
    package initializer for gbswt5.
"""
import transformers

from packaging.version import Version
from functools import partial
from transformers import (
    AutoConfig, AutoModel,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    T5Config, MT5Config, BartModel,
)

from .configuration_gbst5 import GBSWT5Config, GBSWT5OnnxConfig
from .modeling_gbst5 import GBSWT5Model, GBSWT5ForConditionalGeneration, GBSWT5EncoderModel

# GBSWT5 모듈 추가
AutoConfig.register("gbswt5", GBSWT5Config)
AutoModel.register(GBSWT5Config, GBSWT5Model)
AutoModelForSeq2SeqLM.register(GBSWT5Config, GBSWT5ForConditionalGeneration)
#AutoModelForSequenceClassification.register(GBSWT5Config, GBSWT5ForSequenceClassification)


def patch_sentence_transformers_models_Transformer(use_gradient_checkpointing=False):
    """
    monkey patch for sentence_transformers to use GBSWT5EncoderModel.

    compatible with sentence_transformers==2.2.2.
    """
    from sentence_transformers import models

    def _load_model(maybe_self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            maybe_self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            maybe_self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, GBSWT5Config):
            maybe_self._load_gbswt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            maybe_self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config,
                                                              cache_dir=cache_dir, **model_args)
            print(f"Model type: {type(maybe_self.auto_model)}")

    def _load_mt5_model(maybe_self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        maybe_self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, config=config,
                                                                cache_dir=cache_dir, **model_args)

    def _load_gbswt5_model(maybe_self, model_name_or_path, config, cache_dir, **model_args):
        GBSWT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        maybe_self.auto_model = GBSWT5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _forward_base(maybe_self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'],
                          'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features and \
                not isinstance(maybe_self.auto_model, BartModel) and \
                not isinstance(maybe_self.auto_model, transformers.GPTNeoXPreTrainedModel):
            trans_features['token_type_ids'] = features['token_type_ids']

        if isinstance(maybe_self.auto_model, GBSWT5EncoderModel):
            # need to update downsampled attention_mask
            output_states, attention_mask = maybe_self.auto_model(
                **trans_features,
                return_dict=False,
                return_resized_attention_mask=True)
            features.update({'attention_mask': attention_mask})
        else:
            output_states = maybe_self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens,
                         'attention_mask': features['attention_mask']})

        if maybe_self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:
                #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def _forward_wo_cache(maybe_self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'],
                          'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features and \
                not isinstance(maybe_self.auto_model, BartModel) and \
                not isinstance(maybe_self.auto_model, transformers.GPTNeoXPreTrainedModel):
            trans_features['token_type_ids'] = features['token_type_ids']

        if isinstance(maybe_self.auto_model, GBSWT5EncoderModel):
            # need to update downsampled attention_mask
            output_states, attention_mask = maybe_self.auto_model(
                **trans_features,
                return_dict=False,
                return_resized_attention_mask=True,
                use_cache=False,)
            features.update({'attention_mask': attention_mask})
        else:
            output_states = maybe_self.auto_model(**trans_features, return_dict=False,
                                                  use_cache=False,)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens,
                         'attention_mask': features['attention_mask']})

        if maybe_self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:
                #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    models.Transformer._load_model = _load_model
    models.Transformer._load_mt5_model = _load_mt5_model
    models.Transformer._load_gbswt5_model = _load_gbswt5_model
    if use_gradient_checkpointing:
        models.Transformer.forward = _forward_wo_cache
    else:
        models.Transformer.forward = _forward_base

    return models.Transformer
