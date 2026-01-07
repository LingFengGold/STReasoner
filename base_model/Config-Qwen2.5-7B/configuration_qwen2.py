# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Qwen2 model configuration"""

from transformers import PretrainedConfig
from transformers.utils import logging
from typing import *


logger = logging.get_logger(__name__)


class Qwen2TSConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

TINYTIMEMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class TinyTimeMixerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TinyTimeMixerModel`]. It is used to instantiate a
    TinyTimeMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TinyTimeMixer {} architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 64)
            The context/history length for the input sequence.
        patch_length (`int`, *optional*, defaults to 8)
            The patch length for the input sequence.
        num_input_channels (`int`):
            Number of input variates. For Univariate, set it to 1.
        patch_stride (`int`, *optional*, defaults to 8):
            Amount of points to stride. If its value is same as patch_length, we get non-overlapping patches.
        d_model (`int`, *optional*, defaults to 16):
            Hidden feature size of the model.
        prediction_length (`int`, *optional*, defaults to 16)
            Number of time steps to forecast for a forecasting task. Also known as the Forecast Horizon.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 3):
            Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `TinyTimeMixer` backbone. Recommended range is 0.2-0.7
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Mixer Mode. Determines how to process the channels. Allowed values: "common_channel", "mix_channel". In
            "common_channel" mode, we follow Channel-independent modelling with no explicit channel-mixing. Channel
            mixing happens in an implicit manner via shared weights across channels. (preferred first approach) In
            "mix_channel" mode, we follow explicit channel-mixing in addition to patch and feature mixer. (preferred
            approach when channel correlations are very important to model)
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention across patches. This can be enabled when the output of Vanilla TinyTimeMixer with
            gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
            across patches.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
            set to `True`.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. Options `"random"` and `"sincos"` are supported. Works only when
            `use_positional_encoding` is set to `True`
        scaling (`string` or `bool`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean".
        loss (`string`, *optional*, defaults to `"mse"`):
            The loss function for the model. Defaults to mean squared error "mse". Allowed values: ["mse", "mae"]
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, or the default initialization in
            `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        adaptive_patching_levels (`int`, *optional*, defaults to 0):
            If adaptive_patching_levels is i, then we will have i levels with each level having n_layers.
            Level id starts with 0. num_patches at level i will be multipled by (2^i) and num_features at level i will be divided by (2^i).
            For Ex. if adaptive_patching_levels is 3 - then we will have 3 levels:
                level 2: num_features//(2^2), num_patches*(2^2)
                level 1: num_features//(2^1), num_patches*(2^1)
                level 0: num_features//(2^0), num_patches*(2^0)
            adaptive_patching_levels = 1 is same as one level PatchTSMixer. This module gets disabled when adaptive_patching_levels is 0 or neg value. Defaults to 0 (off mode).
        resolution_prefix_tuning (`bool`, *optional*, defaults to `False`):
            Enable if your dataloader has time resolution information as defined in `get_freq_mapping` function in `modelling_tinytimemixer`.
        frequency_token_vocab_size (`int`, *optional*, defaults to 5):
            Vocab size to use when resolution_prefix_tuning is enabled.
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `TinyTimeMixer` head.
        prediction_channel_indices (`list`, *optional*):
            List of channel indices to forecast. If None, forecast all channels. Target data is expected to have all
            channels and we explicitly filter the channels in prediction and target before loss computation. Please provide the indices
            in sorted ascending order.
        decoder_num_layers (`int`, *optional*, defaults to 8):
            Number of layers to use in decoder
        decoder_d_model(`int`, *optional*, defaults to 16):
            Defines the hidden feature size of the decoder.
        decoder_adaptive_patching_levels (`int`, *optional*, defaults to 0):
            Adaptive Patching levels for decoder. Preferable to set it to 0 for decoder to keep it light weight.
        decoder_raw_residual (`bool`, *optional*, defaults to `False`):
            Flag to enable merging of raw embedding with encoder embedding for decoder input. Defaults to False.
        decoder_mode (`string`, *optional*, defaults to `"common_channel"`):
            Decoder channel mode. Use `"common_channel" for channel-independent modelling and `"mix_channel"` for channel-mixing modelling
        use_decoder (`bool`, *optional*, defaults to `True`):
            Enable to use decoder.
        prediction_filter_length (`int`,*optional*, defaults to None):
            Actual length in the prediction output to use for loss calculations.


    Example:

    ```python
    >>> from transformers import TinyTimeMixerConfig, TinyTimeMixerModel

    >>> # Initializing a default TinyTimeMixer configuration
    >>> configuration = TinyTimeMixerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = TinyTimeMixerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tinytimemixer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 64,
        patch_length: int = 8,
        num_input_channels: int = 1,
        prediction_length: int = 16,
        patch_stride: int = 8,
        prediction_channel_indices: Optional[list] = None,
        # General model configuration
        d_model: int = 16,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: Optional[Union[str, bool]] = "std",
        loss: str = "mse",
        init_std: float = 0.02,
        post_init: bool = False,
        norm_eps: float = 1e-5,
        adaptive_patching_levels: int = 0,
        resolution_prefix_tuning: bool = False,
        frequency_token_vocab_size: int = 5,
        # General head configuration
        head_dropout: float = 0.2,
        # decoder parameters
        decoder_num_layers: int = 8,
        decoder_d_model: int = 8,
        decoder_adaptive_patching_levels: int = 0,
        decoder_raw_residual: bool = False,
        decoder_mode: str = "common_channel",
        use_decoder: bool = True,
        # prediction length filtering
        prediction_filter_length: Optional[int] = None,
        **kwargs,
    ):
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout

        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.prediction_length = prediction_length
        self.prediction_channel_indices = prediction_channel_indices
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.loss = loss
        self.norm_eps = norm_eps

        self.use_decoder = use_decoder

        self.adaptive_patching_levels = adaptive_patching_levels
        self.resolution_prefix_tuning = resolution_prefix_tuning
        self.decoder_num_layers = decoder_num_layers
        self.decoder_adaptive_patching_levels = decoder_adaptive_patching_levels
        self.decoder_raw_residual = decoder_raw_residual
        self.decoder_mode = decoder_mode
        self.frequency_token_vocab_size = frequency_token_vocab_size
        self.d_model = d_model
        self.patch_stride = patch_stride
        self.decoder_d_model = decoder_d_model
        self.init_processing = False
        self.prediction_filter_length = prediction_filter_length

        super().__init__(**kwargs)

    def check_and_init_preprocessing(self):
        self.init_processing = True

        if not hasattr(self, "num_patches"):
            self.num_patches = (
                max(self.context_length, self.patch_length) - self.patch_length
            ) // self.patch_stride + 1

            if self.resolution_prefix_tuning:
                self.num_patches += 1

        if self.prediction_filter_length is not None:
            if self.prediction_filter_length > self.prediction_length or self.prediction_filter_length <= 0:
                raise ValueError("prediction_filter_length should be positive and less than prediction_length")

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()
