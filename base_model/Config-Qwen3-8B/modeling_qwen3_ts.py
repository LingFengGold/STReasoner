# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Union

import torch
from torch import nn
from dataclasses import dataclass

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3Model
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
try:
    from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, logging, ModelOutput
except ImportError:
    from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, ModelOutput
from .configuration_qwen3_ts import Qwen3TSConfig
from typing import Any, Dict


logger = logging.get_logger(__name__)


########################MLP TS Embedding#####################
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, config):
        super(TimeSeriesEmbedding, self).__init__()
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.num_features = config['num_features']
        self.max_sequence_length = config['max_sequence_length']  # Maximum time series length
        self.use_position_embedding = config.get('use_position_embedding', False)
        self.use_position_idx = config.get('use_position_idx', False)
        self.use_layer_norm = config.get('use_layer_norm', False)
        self.embedding_dim = config.get('embedding_dim', 16)  # Embedding dimension
        
        if self.use_position_embedding:
            # Extended vocabulary: [0, max_sequence_length) for real positions, max_sequence_length for padding
            self.position_embedding = nn.Embedding(self.max_sequence_length + 1, self.embedding_dim)
            self.padding_idx = self.max_sequence_length  # Special index for padding
            input_size = 1 * self.patch_size + self.embedding_dim * self.patch_size
        elif self.use_position_idx:
            input_size = 2 * self.patch_size
        else:
            input_size = 1 * self.patch_size
        
        # Build MLP layers
        layers = []
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(input_size, self.hidden_size))
            layers.append(nn.GELU())
            input_size = self.hidden_size

        layers.append(nn.Linear(input_size, self.hidden_size))
        if self.use_layer_norm:
            layers.append(nn.LayerNorm(self.hidden_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)

        # Extract mask and calculate valid lengths
        mask = x[:, :, -1].long()
        valid_lengths = mask.sum(dim=1).long()
        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size

        patches_list = []
        # Collect position indices for batch embedding lookup
        all_position_indices = []
        patch_info_list = []  # Store metadata for each patch group
        
        for i in range(batch_size):
            vl = valid_lengths[i].item()
            pc = patch_cnt[i].item()
            if pc == 0:
                continue
            
            # Extract time series data (excluding mask)
            xi = x[i, :vl, :1]  # Time-series data
            total_padded_length = pc * self.patch_size
            padding_length = total_padded_length - vl
            
            # Create position indices: real positions for actual data, special index for padding
            position_indices = torch.arange(vl, device=x.device)
            
            if padding_length > 0:
                # Pad with last value
                last_value = xi[-1:, :]
                padding = last_value.repeat(padding_length, 1)
                xi = torch.cat([xi, padding], dim=0)
                
                # Use special padding index for padding positions
                padding_positions = torch.full((padding_length,), self.padding_idx, device=x.device)
                position_indices = torch.cat([position_indices, padding_positions], dim=0)

            # Reshape to patches
            xi = xi.reshape(pc, self.patch_size)  # (num_patches, patch_size)
            position_indices = position_indices.reshape(pc, self.patch_size)  # (num_patches, patch_size)

            if self.use_position_embedding:
                # Collect position indices instead of calling embedding immediately
                all_position_indices.append(position_indices)
                patch_info_list.append({
                    'xi': xi,
                    'pc': pc,
                    'sample_idx': i
                })
            elif self.use_position_idx:
                # Normalize position indices
                pos_indices = torch.arange(vl, device=x.device).unsqueeze(1)
                pos_indices = pos_indices / max(1, valid_lengths.max().item() - 1)
                if padding_length > 0:
                    # Use -1 for padding positions
                    padding_indices = torch.full((padding_length, 1), -1, device=x.device)
                    pos_indices = torch.cat([pos_indices, padding_indices], dim=0)
                # Combine time series data with position indices
                xi_combined = torch.cat([xi.reshape(-1, 1), pos_indices], dim=1)
                patch_input = xi_combined.reshape(pc, self.patch_size * 2)
                patches_list.append(patch_input)
            else:
                # No position embedding, use raw patches
                patch_input = xi
                patches_list.append(patch_input)

        # Batch process position embeddings if needed
        if self.use_position_embedding and all_position_indices:
            # Concatenate all position indices for batch embedding lookup
            batch_position_indices = torch.cat(all_position_indices, dim=0)
            # print(f"{x.shape=}, {x.device=}, {len(all_position_indices)=}, {batch_position_indices=}")
            batch_pos_emb = self.position_embedding(batch_position_indices)  # Single embedding call
            
            # Split embeddings back and create patch inputs
            emb_start_idx = 0
            for patch_info in patch_info_list:
                xi = patch_info['xi']
                pc = patch_info['pc']
                
                # Extract corresponding embeddings
                pos_emb = batch_pos_emb[emb_start_idx:emb_start_idx + pc]
                emb_start_idx += pc
                
                # Flatten and concatenate
                xi = xi.unsqueeze(-1)  # (num_patches, patch_size, 1)
                patch_input = torch.cat([
                    xi.flatten(1),  # (num_patches, patch_size)
                    pos_emb.flatten(1)  # (num_patches, patch_size * embedding_dim)
                ], dim=1)
                patches_list.append(patch_input)

        # Process all patches through MLP
        if patches_list:
            x_patches = torch.cat(patches_list, dim=0)
            x = self.mlp(x_patches)
        else:
            # Handle empty case
            x = torch.empty(0, self.hidden_size, device=x.device)

        return x, patch_cnt


@dataclass
class Qwen3TSCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Output type of Qwen3TSForCausalLM that includes additional fields for timeseries processing.
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The attention mask used in the forward pass, potentially expanded to accommodate timeseries patches.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.
        new_token_positions (`torch.LongTensor` of shape `(batch_size, num_new_tokens)`, *optional*):
            Positions where new tokens (not from timeseries) are located in the expanded sequence.
    """
    attention_mask: Optional[torch.FloatTensor] = None
    labels: Optional[torch.LongTensor] = None
    new_token_positions: Optional[torch.LongTensor] = None

try:
    LossKwargs
    _BaseKwargs = LossKwargs
except NameError:
    _BaseKwargs = TransformersKwargs


class KwargsForCausalLM(FlashAttentionKwargs, _BaseKwargs): ...


class Qwen3TSGenerationMixin(GenerationMixin):
    """
    Generation mixin for Qwen3 models with timeseries support.
    
    This mixin handles the special case where timeseries embeddings expand the sequence length
    during the first forward pass, requiring special attention mask management.
    """
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        timeseries: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation with timeseries support.
        
        Timeseries are only processed during the first forward pass. In subsequent
        generation steps, they are already embedded in the past_key_values.
        """
        # Check if we have timeseries data
        has_ts = timeseries is not None and len(timeseries) > 0
        
        # Handle timeseries for generation with past_key_values
        if has_ts and past_key_values is not None:
            # Get the number of tokens already processed
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.seen_tokens
            else:
                past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            
            # If we have processed tokens, timeseries have already been embedded
            if past_length > 0:
                # Only keep the last token for next token prediction
                input_ids = input_ids[:, -1:]
                # Clear timeseries as they've been processed
                timeseries = None
                has_ts = False
        
        # Call parent's prepare_inputs_for_generation to handle all standard logic
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )
        
        # Add timeseries to model inputs
        model_inputs["timeseries"] = timeseries
        
        return model_inputs
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        """
        Update model keyword arguments for generation, handling attention mask from outputs.
        
        This is necessary because timeseries processing can expand the sequence length,
        and we need to use the expanded attention_mask from the model outputs.
        """
        # Handle special case: use attention_mask from outputs if available
        # This is crucial for timeseries models where the sequence length is expanded
        if hasattr(outputs, "attention_mask") and outputs.attention_mask is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask
        
        # Call parent's implementation for standard updates
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        
        return model_kwargs
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        timeseries: Optional[torch.FloatTensor] = None,
        generation_config=None,
        **kwargs,
    ):
        """
        Generate sequences with timeseries support.
        
        Args:
            inputs: Input token ids
            timeseries: Optional timeseries data to be processed in the first forward pass
            generation_config: Generation configuration
            **kwargs: Additional keyword arguments for generation
        """
        # Add timeseries to kwargs if provided
        if timeseries is not None:
            kwargs["timeseries"] = timeseries
        
        # Call parent's generate method
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            **kwargs,
        )
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]) -> None:
        """
        Validate model kwargs, allowing timeseries as a valid argument.
        """
        # Remove timeseries from model_kwargs temporarily for validation
        timeseries = model_kwargs.pop("timeseries", None)
        
        # Call parent's validation
        super()._validate_model_kwargs(model_kwargs)
        
        # Restore timeseries
        if timeseries is not None:
            model_kwargs["timeseries"] = timeseries

class Qwen3TSPreTrainedModel(Qwen3PreTrainedModel):
    config_class = Qwen3TSConfig

@auto_docstring
class Qwen3TSForCausalLM(Qwen3TSPreTrainedModel, Qwen3TSGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # TS embedding
        self.ts_encoder = TimeSeriesEmbedding(config.ts)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def _merge_input_ids_with_time_series_features(self, time_series_features, inputs_embeds, input_ids, attention_mask, labels, patch_cnt):
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)
        left_padding = False
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                left_padding = False
            else:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
        else:
            if _left_padding and not _right_padding:
                left_padding = True
            else:
                left_padding = False

        # 1. Create a mask to know where special time series tokens are
        special_ts_token_mask_start = input_ids == self.config.ts_token_start_index
        special_ts_token_mask_end = input_ids == self.config.ts_token_end_index
        special_ts_token_mask = special_ts_token_mask_start | special_ts_token_mask_end

        # 2. Calculate patch count
        num_special_ts_tokens = torch.sum(special_ts_token_mask_start, dim=-1)
        total_time_steps, embed_dim = time_series_features.shape

        # Correctly calculate the total number of patches per batch
        patch_index = 0
        num_total_patches = torch.zeros(batch_size, dtype=patch_cnt.dtype, device=patch_cnt.device)
        special_ts_token_mask_start_nonzero = special_ts_token_mask_start.nonzero()
        special_ts_token_mask_start_with_size = special_ts_token_mask_start.clone().long()

        attn_mask_cnt = attention_mask.sum(dim=-1)
        for i in range(batch_size):
            num_ts_in_batch = num_special_ts_tokens[i]
            num_total_patches[i] = patch_cnt[patch_index : patch_index + num_ts_in_batch].sum() - 2 * num_ts_in_batch
            for idx in range(patch_index, patch_index + num_ts_in_batch):
                b_idx, pos = special_ts_token_mask_start_nonzero[idx]
                special_ts_token_mask_start_with_size[b_idx, pos] *= (patch_cnt[idx].item() - 2)
            patch_index += num_ts_in_batch
            attn_mask_cnt[i] += num_total_patches[i].item()

        # 3. Embeding length
        max_embed_dim = sequence_length + num_total_patches.max()

        # 4. Non ts tokens
        batch_indices, non_ts_indices = torch.where(~special_ts_token_mask)
        attn_batch_indices, attn_indices = torch.where(attention_mask == 1)

        # 5. Text token in final text positions
        new_token_positions = torch.cumsum((special_ts_token_mask_start_with_size + 1), dim=-1) - 1

        # nb_ts_pad
        nb_ts_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_ts_pad[:, None]

        text_to_overwrite = new_token_positions[batch_indices, non_ts_indices]

        # 6. Final embedding and attention masks
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        final_attention_mask = torch.zeros(batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device)
        for i in range(attention_mask.size(0)):
            if left_padding:
                final_attention_mask[i, max_embed_dim - attn_mask_cnt[i] :] = 1
            else:
                final_attention_mask[i, : attn_mask_cnt[i]] = 1

        final_labels = None
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )

        target_device = inputs_embeds.device
        batch_indices, non_ts_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_ts_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 7. Move embedding and labels to final positions
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_ts_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_ts_indices]

        # 8. Move time series to final positions
        ts_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        ts_to_overwrite[batch_indices, text_to_overwrite] = False

        reversed_cumsum = ts_to_overwrite.flip(dims=[-1]).cumsum(-1).flip(dims=[-1]) - 1
        ts_to_overwrite &= reversed_cumsum >= nb_ts_pad[:, None].to(target_device)

        # Check that the number of time series tokens is correct
        if ts_to_overwrite.sum() != time_series_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of time series tokens is {torch.sum(special_ts_token_mask_start)} while"
                f" the number of time series given to the model is {len(patch_cnt)}. This prevents correct indexing and breaks batch generation."
            )
        final_embedding[ts_to_overwrite] = time_series_features.contiguous().reshape(-1, embed_dim).to(target_device)
        # if str(input_ids.device) == 'cuda:0':
        #     print(f"[EMBED] {final_embedding[ts_to_overwrite][0]=}")

        # 9. Calculate position ids
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        if position_ids.size(-1) < input_ids.size(-1):
            position_ids = position_ids[:, -input_ids.size(-1) :]

        # 10. Move attention mask to final positions
        # print(f"{type(input_ids)=}, {input_ids.shape=}, {self.config.pad_token_id=}")
        pad_batch_indices, pad_indices = torch.where(input_ids == self.config.pad_token_id)
        if len(pad_batch_indices) > 0:
            indices_to_mask = new_token_positions[pad_batch_indices, pad_indices]
            final_embedding[pad_batch_indices, indices_to_mask] = 0

        # 11. Post-process new_token_positions (set -1 for padding positions)
        new_token_positions = new_token_positions.masked_fill(attention_mask == 0, -1)

        return final_embedding, final_attention_mask, position_ids, final_labels, new_token_positions

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        timeseries: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Qwen3TSCausalLMOutputWithPast:
        r"""
        Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the attention blocks).
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        timeseries (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size)`, *optional*):
            Timeseries data to be encoded and merged with text embeddings.
    
    Returns:
        [`Qwen3TSCausalLMOutputWithPast`] or `tuple(torch.FloatTensor)`:
        The model outputs with potential timeseries-expanded attention mask.
        """

        # if input_ids is not None and timeseries is not None:
        #     # Print the input ts
        #     print("=================================================================")
        #     print("Timeseries shape:", timeseries.shape)
        #     print("=================================================================\n\n")
        # else:
        #     print("Time series is None!!!!")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if timeseries is not None and timeseries.shape[0] > 0:
                # use_cache = False
                # print(f"timeseries shape: {timeseries.shape=}, {input_ids.shape=}")
                ts_features, patch_cnt = self.ts_encoder(timeseries)
                inputs_embeds = inputs_embeds.to(ts_features.dtype)
                # if str(input_ids.device) == 'cuda:0':
                #     print(f"-------------------------------------------------------\n[before] {input_ids.shape=}, timeseries={timeseries.shape if timeseries is not None else None}, {attention_mask.sum(-1)=}")
                inputs_embeds, attention_mask, position_ids, labels, new_token_positions = self._merge_input_ids_with_time_series_features(
                    ts_features, inputs_embeds, input_ids, attention_mask, labels, patch_cnt
                )
                # print(f"{inputs_embeds.shape=}, {attention_mask.shape=}, {position_ids.shape=}, {labels.shape=}")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # if str(input_ids.device) == 'cuda:0':
        #     print(f"[after] {inputs_embeds.shape=}, {attention_mask.sum(-1)=}, {position_ids.max(-1)=}, {cache_position=}")
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        # print(f"{logits.shape=}, {hidden_states.shape=}, {slice_indices=}, {logits_to_keep=}, labels={(labels != self.config.ignore_index).sum() if labels is not None else None}, {loss=}, {torch.sum(torch.isnan(logits))=}, {torch.sum(torch.isinf(logits))=}, {torch.sum(torch.isnan(hidden_states))=}, {torch.sum(torch.isinf(hidden_states))=}")

        return Qwen3TSCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            labels=labels,
            new_token_positions=new_token_positions if timeseries is not None and timeseries.shape[0] > 0 else None,
        )


__all__ = [
    "Qwen3TSForCausalLM"
]