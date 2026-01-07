# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def remove_graph_structure(messages: Union[str, list[dict[str, Any]]]) -> Union[str, list[dict[str, Any]]]:
    """
    Remove 'Graph Structure: ...' until 'please analyze' from prompt text.

    Designed for simple cases: either a raw string or a single-chat message
    list like [{"role": "user", "content": prompt_str}]. Other inputs are
    returned unchanged.
    """
    pattern = r"Graph Structure:.*?(?=please analyze)"
    strip = lambda txt: re.sub(pattern, "", txt, flags=re.DOTALL | re.IGNORECASE)

    if isinstance(messages, str):
        return strip(messages)

    if isinstance(messages, list) and messages:
        msg = messages[0].copy()
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = strip(content)
        return [msg]

    return messages

def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        ts_key: str = "timeseries",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        enable_spatial_reward: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.ts_key = ts_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.enable_spatial_reward = enable_spatial_reward

        # Support multiple datasets by allowing comma-separated paths in `data_path`
        data_paths = [p.strip() for p in data_path.split(",")] if "," in data_path else [data_path]
        loaded_datasets = []

        for single_path in data_paths:
            if "@" in single_path:
                single_path, data_split = single_path.split("@")
            else:
                data_split = "train"

            if os.path.isdir(single_path):
                # when we use dataset builder, we should always refer to the train split
                file_type = os.path.splitext(os.listdir(single_path)[0])[-1][1:].replace("jsonl", "json")
                ds = load_dataset(file_type, data_dir=single_path, split=data_split)
            elif os.path.isfile(single_path):
                file_type = os.path.splitext(single_path)[-1][1:].replace("jsonl", "json")
                ds = load_dataset(file_type, data_files=single_path, split=data_split)
            else:
                # load remote dataset from huggingface hub
                ds = load_dataset(single_path, split=data_split)

            loaded_datasets.append(ds)

        if len(loaded_datasets) == 1:
            self.dataset = loaded_datasets[0]
        else:
            self.dataset = concatenate_datasets(loaded_datasets)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.ts_key in example:
            return [{"role": "user", "content": prompt_str}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.ts_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            timeseries = example[self.ts_key]
            model_inputs = self.processor(timeseries=timeseries, text=[prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            # [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'The area $A$ of the shaded region is given. Find $x$. $A = 66$ cm$^2$ . You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'}]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # <|im_start|>system                                                                                                                        
            # You are a helpful assistant.<|im_end|>                                                                                                                                                                                                                    
            # <|im_start|>user                                                                                                                                 
            # <|vision_start|><|image_pad|><|vision_end|>The area $A$ of the shaded region is given. Find $x$. $A = 66$ cm$^2$ . You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.<|im_end|>
            # <|im_start|>assistant       
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        elif self.ts_key in example:  
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            timeseries = example[self.ts_key]
            model_inputs = self.processor(timeseries=timeseries, text=[prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"timeseries": timeseries}
            
            # Generate no-graph version for spatial reward comparison
            if self.enable_spatial_reward:
                messages_no_graph = remove_graph_structure(messages)
                prompt_no_graph = self.processor.apply_chat_template(
                    messages_no_graph, add_generation_prompt=True, tokenize=False
                )
                model_inputs_no_graph = self.processor(
                    timeseries=timeseries, text=[prompt_no_graph], add_special_tokens=False, return_tensors="pt"
                )
                input_ids_no_graph = model_inputs_no_graph.pop("input_ids")[0]
                attention_mask_no_graph = model_inputs_no_graph.pop("attention_mask")[0]
                example["input_ids_no_graph"] = input_ids_no_graph
                example["attention_mask_no_graph"] = attention_mask_no_graph
                example["prompt_no_graph"] = prompt_no_graph
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.ts_key in example:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
        elif self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        
        # Process no-graph version if spatial reward is enabled
        if self.enable_spatial_reward and self.ts_key in example:
            input_ids_no_graph = example["input_ids_no_graph"]
            attention_mask_no_graph = example["attention_mask_no_graph"]
            prompt_no_graph = example["prompt_no_graph"]
            
            # Compute position_ids for no-graph version
            position_ids_no_graph = torch.clip(attention_mask_no_graph.cumsum(dim=0) - 1, min=0, max=None)
            
            # Apply postprocess_data to no-graph version
            input_ids_no_graph, attention_mask_no_graph, position_ids_no_graph = VF.postprocess_data(
                input_ids=input_ids_no_graph,
                attention_mask=attention_mask_no_graph,
                position_ids=position_ids_no_graph,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            
            # Compute raw_prompt_ids for no-graph version
            raw_prompt_ids_no_graph = self.tokenizer.encode(prompt_no_graph, add_special_tokens=False)
            if len(raw_prompt_ids_no_graph) > self.max_prompt_length:
                if self.truncation == "left":
                    raw_prompt_ids_no_graph = raw_prompt_ids_no_graph[-self.max_prompt_length:]
                elif self.truncation == "right":
                    raw_prompt_ids_no_graph = raw_prompt_ids_no_graph[:self.max_prompt_length]
                elif self.truncation == "error":
                    raise RuntimeError(f"No-graph prompt length {len(raw_prompt_ids_no_graph)} is longer than {self.max_prompt_length}.")
            
            example["input_ids_no_graph"] = input_ids_no_graph
            example["attention_mask_no_graph"] = attention_mask_no_graph
            example["position_ids_no_graph"] = position_ids_no_graph
            example["raw_prompt_ids_no_graph"] = raw_prompt_ids_no_graph
            del example["prompt_no_graph"]  # Clean up temporary field
        
        return example
