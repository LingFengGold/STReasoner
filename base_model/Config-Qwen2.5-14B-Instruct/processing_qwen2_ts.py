# coding=utf-8
# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import List, Union, Tuple, Optional
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy

def sp_encoding(timeseries: np.ndarray, eots_token: bool = True) -> Tuple[np.ndarray, str, dict]:
    """
    Encodes a time series with scalar normalization.

    Args:
        timeseries (np.ndarray): The raw time series data (1D or 2D).

    Returns:
        result_timeseries (np.ndarray): The encoded time series, shape [seq_len, 1].
        prompt (str): The placeholder string with offset and scaling info.
        metadata (dict): Metadata containing the offset and scaling factor.
    """
    timeseries = np.array(timeseries)
    mean = np.mean(timeseries)
    scaled_timeseries = timeseries - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled_timeseries) >= 3.0):
        scale_factor = np.max(np.abs(scaled_timeseries)) / 3.0
        scaled_timeseries /= scale_factor

    prompt = f"[offset={-mean:.4f}|scaling={scale_factor:.4f}|length={len(timeseries)}|max={max(timeseries):.4f}|min={min(timeseries):.4f}|left={timeseries[0]:.4f}|right={timeseries[-1]:.4f}]<ts>"
    if eots_token:
        prompt += '<ts/>'

    result_timeseries = np.stack([scaled_timeseries, np.ones_like(scaled_timeseries)], axis=-1).reshape(-1, 1)

    return result_timeseries, prompt, {"offset": float(-mean), "scale_factor": float(scale_factor)}

class Qwen2TSProcessor(ProcessorMixin):
    """
    A processor for ChatTS that integrates text prompt processing and time series encoding.
    """

    attributes = ["tokenizer"]
    feature_extractor_class = None  # You can add a feature extractor if needed
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, chat_template=None, **kwargs):
        """
        Args:
            tokenizer: An optional tokenizer to process text prompts.
        """
        if chat_template is None and tokenizer is not None and tokenizer.chat_template is not None:
            chat_template = tokenizer.chat_template
        self.chat_template = chat_template

        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        text: Union[str, List[str]],
        timeseries: Optional[List[List[np.ndarray]]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        padding_side: str = 'left',
        vllm_flag: bool = False,
        tokenize: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Encodes a prompt and its associated time series.

        Args:
            prompt (List[str]): The input prompt containing <ts><ts/> placeholders.
            timeseries (List[np.ndarray]): A list of time series matched to placeholders in the prompt.
            padding (bool or str or PaddingStrategy, optional): Passed to the tokenizer for text padding.
            return_tensors (str, optional): "pt" to return PyTorch tensors; None to return NumPy arrays.
            **kwargs: Additional tokenizer parameters.

        Returns:
            BatchFeature: Contains processed prompt, encoded time series, and tokenizer outputs.
        """
        if type(text) == str:
            text = [text]
        if timeseries is None:
            timeseries = []

        reconstructed_prompts = []
        concatenated_ts = None
        ts_tokens = []

        if vllm_flag:
            # All prompt modifications have to be done inside of the vLLM
            # to work correctly with its caching mechanism.
            reconstructed_prompts = text
            
            # Process timeseries data
            encoded_ts_arrays = []
            for ts in timeseries:
                # Get the normalized data and prompt text
                encoded_ts, ts_prompt, _ = sp_encoding(ts, eots_token=False)
                # Tokenize the ts_prompt and add to the tokens list
                if self.tokenizer is not None:
                    tokens = self.tokenizer.encode(ts_prompt, add_special_tokens=False)
                    ts_tokens.append(tokens)
                encoded_ts_arrays.append(encoded_ts[None, ...])
        else:
            encoded_ts_arrays = []
            total_ts_cnt = 0
            for idx, prompt in enumerate(text):
                # Split prompt by <ts><ts/> placeholders
                last_ts_cnt = total_ts_cnt
                prompt_segments = prompt.split("<ts><ts/>")
                total_ts_cnt = total_ts_cnt + len(prompt_segments) - 1

                # Encode each time series and rebuild the prompt
                reconstructed_prompt = prompt_segments[0]

                for i, ts in enumerate(timeseries[last_ts_cnt:total_ts_cnt]):
                    encoded_ts, ts_prompt, _ = sp_encoding(ts, eots_token=not vllm_flag)
                    reconstructed_prompt += ts_prompt + prompt_segments[i + 1]
                    # Ensure time series shape [1, seq_len, feature_dim] for batch concatenation
                    encoded_ts_arrays.append(encoded_ts[None, ...])

                reconstructed_prompts.append(reconstructed_prompt)

            if len(timeseries) != len(encoded_ts_arrays):
                raise ValueError(
                    f"Mismatch between <ts><ts/> placeholders ({total_ts_cnt}) "
                    f"and time series ({len(encoded_ts_arrays)})."
                )

            if len(encoded_ts_arrays) > 0:
                # Pad time series to the same length
                max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
                padded_ts_arrays = [
                    np.pad(ts, ((0, 0), (0, max_length - ts.shape[1]), (0, 0)), mode="constant", constant_values=0.0)
                    for ts in encoded_ts_arrays
                ]
                concatenated_ts = np.concatenate(padded_ts_arrays, axis=0)  # Shape: [batch_size, max_length, feature_dim]
                
                # Convert to torch
                concatenated_ts = torch.from_numpy(concatenated_ts).half()

        # Tokenize the processed prompt
        tokenizer_outputs = {}
        if tokenize and self.tokenizer is not None:
            tokenizer_outputs = self.tokenizer(reconstructed_prompts, padding=padding, padding_side=padding_side, **kwargs)
        else:
            tokenizer_outputs = {"text": reconstructed_prompts}

        # Create the final output
        outputs = tokenizer_outputs
        if vllm_flag:
            outputs["timeseries"] = zip(ts_tokens, encoded_ts_arrays)
        elif concatenated_ts is not None:
            outputs["timeseries"] = concatenated_ts

        return BatchFeature(data=outputs)

    def encode_timeseries(
        self,
        timeseries: Optional[List[List[np.ndarray]]] = None,
    ) -> np.ndarray:
        if timeseries is None:
            timeseries = []

        concatenated_ts = None
        encoded_ts_arrays = []

        for i, ts in enumerate(timeseries):
            encoded_ts, _, _ = sp_encoding(ts)
            # Ensure time series shape [1, seq_len, feature_dim] for batch concatenation
            encoded_ts_arrays.append(encoded_ts[None, ...])

        if len(encoded_ts_arrays) > 0:
            # Pad time series to the same length
            max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
            padded_ts_arrays = [
                np.pad(ts, ((0, 0), (0, max_length - ts.shape[1]), (0, 0)), mode="constant", constant_values=0.0)
                for ts in encoded_ts_arrays
            ]
            concatenated_ts = np.concatenate(padded_ts_arrays, axis=0)  # Shape: [batch_size, max_length, feature_dim]
            
            # Convert to torch
            concatenated_ts = torch.from_numpy(concatenated_ts).half()

        return concatenated_ts

    @property
    def model_input_names(self):
        """
        Define the input names expected by the model.
        """
        tokenizer_input_names = []
        if self.tokenizer and hasattr(self.tokenizer, "model_input_names"):
            tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(["processed_prompt", "time_series"] + tokenizer_input_names))

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)