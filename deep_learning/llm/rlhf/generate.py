"""
Script for generating answers/completions on ultrafeedback dataset with
models or models + adapters
"""
import os
import torch
import numpy as np
import pandas as pd
import transformers
import pytorch_lightning as pl
from peft import PeftModel
from datasets import load_dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from typing import Any, Dict, List, Optional, Union
from transformers.data.data_collator import DataCollatorMixin
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForGeneration(DataCollatorMixin):
    """
    tokenize raw text (prompt) as well as padding while forming a batch for data loader.
    """

    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int = 512
    padding: Union[bool, str, PaddingStrategy] = True
    return_tensors: str = "pt"
    prompt_col_name: str = "prompt"

    def __post_init__(self):
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(
        self, features: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:

        prompts = [feature[self.prompt_col_name] for feature in features]
        tokenized_text = self.tokenizer(
            prompts,
            padding=self.padding,
            max_length=self.max_seq_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
        )

        batch = {
            "prompts": prompts,
            "input_ids": tokenized_text["input_ids"],
            "attention_mask": tokenized_text["attention_mask"],
        }
        return batch


class LLMGenerateLightningModule(pl.LightningModule):
    """
    Generate responses from LLM. Expects input prompts, tokenized input_ids, attention_mask
    """

    def __init__(
        self,
        pretrained_model_name_or_path,
        generation_config,
        prediction_config,
        adapter_path=None,
        cache_dir="/data",
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, cache_dir=cache_dir
        )
        if adapter_path:
            peft_model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = peft_model.merge_and_unload()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, padding_side="left", cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_config = generation_config
        self._setup_prediction(prediction_config)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        prompts = batch["prompts"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        responses = self.generate(input_ids, attention_mask)

        prediction_output = {
            "prompts": prompts,
            "responses": responses,
        }
        self.prediction_outputs.append(prediction_output)
        return prediction_output

    def generate(self, input_ids, attention_mask):
        model_output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config
        )
        # crop input prompt from generated response
        input_seq_length = input_ids.shape[-1]
        model_output_answer_only = model_output[:, input_seq_length:]
        responses = self.tokenizer.batch_decode(model_output_answer_only, skip_special_tokens=True)
        return responses

    def _setup_prediction(self, prediction_config):
        if prediction_config:
            self.prediction_outputs = []
            self._prediction_partition_idx = 0
            self.prediction_partition_format = prediction_config["prediction_partition_format"]
            self.prediction_output_path = prediction_config["prediction_output_path"]
            self.prediction_accumulation_steps = prediction_config.get("prediction_accumulation_steps", 100)

    def _save_prediction_outputs(self):
        if self.prediction_output_path:
            data = {field: [] for field in self.prediction_outputs[0]}
            for prediction_output in self.prediction_outputs:
                for field in data:
                    data[field].extend(prediction_output[field])

            partition_file_name = self.prediction_partition_format.format(
                rank=self.global_rank, partition=self._prediction_partition_idx
            )
            formatted_output_path = os.path.join(
                self.prediction_output_path, partition_file_name
            )

            # saves prediction batch locally via pandas data frame
            df_prediction_outputs = pd.DataFrame.from_dict(data)
            os.makedirs(self.prediction_output_path, exist_ok=True)
            df_prediction_outputs.to_parquet(formatted_output_path, index=False)

        self._prediction_partition_idx += 1
        self.prediction_outputs.clear()

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        if len(self.prediction_outputs) == self.prediction_accumulation_steps:
            self._save_prediction_outputs()

    def on_predict_epoch_end(self):
        if len(self.prediction_outputs) > 0:
            self._save_prediction_outputs()



if __name__ == "__main__":
    pretrained_model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    # generate response from a instruct model, versus a model trained
    # on ultra feedback dataset using LoRA
    adapter_path = None
    prediction_output_path = "prediction_instruction_3B_model"
    # adapter_path = "dpo_model_v7"
    # prediction_output_path = "prediction_dpo_model_v7"

    dataset = load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        split="train",
        verification_mode="no_checks",
        cache_dir="/data"
    )
    dataset_dict = dataset.train_test_split(test_size=0.001, seed=54321)
    examples = dataset_dict["test"]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    data_collator = DataCollatorForGeneration(tokenizer)
    data_loader = DataLoader(examples, batch_size=2, num_workers=2, collate_fn=data_collator)

    generation_config = GenerationConfig(
        max_new_tokens=250
    )
    llm_generate_module = LLMGenerateLightningModule(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        adapter_path=adapter_path,
        generation_config=generation_config,
        prediction_config={
            "prediction_output_path": prediction_output_path,
            "prediction_partition_format": "rank-{rank:02d}-partition-{partition:06d}.parquet"
        }
    )
    trainer = pl.Trainer()
    trainer.predict(llm_generate_module, data_loader)