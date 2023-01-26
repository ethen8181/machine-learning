"""
Run a seq2seq Marian translation model evaluation on wmt16 dataset.
"""
import os
import torch
import random
import evaluate
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from translation_utils import download_file, create_translation_data


@dataclass
class Config:
    cache_dir: str = "./translation"
    data_dir: str = os.path.join(cache_dir, "wmt16")
    source_lang: str = 'de'
    target_lang: str = 'en'    
    
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    max_source_length: int = 128
    max_target_length: int = 128

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_checkpoint: str = "Helsinki-NLP/opus-mt-de-en"

    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_checkpoint,
            cache_dir=self.cache_dir
        )
        print('# of parameters: ', self.model.num_parameters())


def batch_tokenize_fn(examples):
    """
    Generate the input_ids and labels field for huggingface dataset/dataset dict.

    Truncation is enabled where we cap the sentence to the max length. Padding will be done later
    in a data collator, so we pad examples to the longest length within a mini-batch and not
    the whole dataset.
    """
    sources = examples[config.source_lang]
    targets = examples[config.target_lang]
    model_inputs = config.tokenizer(sources, max_length=config.max_source_length, truncation=True)

    # setup the tokenizer for targets,
    # huggingface expects the target tokenized ids to be stored in the labels field
    with config.tokenizer.as_target_tokenizer():
        labels = config.tokenizer(targets, max_length=config.max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    """
    note: we can run trainer.predict on our eval/test dataset to see what a sample
    eval_pred object would look like when implementing custom compute metrics function
    """
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = config.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, config.tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = config.tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    score = sacrebleu_score.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result["sacrebleu"] = score["score"]
    return {k: round(v, 4) for k, v in result.items()}


def create_wmt16_data_files(config: Config):
    # files are downloaded from
    # http://www.statmt.org/wmt16/multimodal-task.html
    urls = [
        'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
        'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
        'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz'
    ]

    for url in urls:
        download_file(url, config.data_dir)

    data_files = {}
    for split in ["train", "val", "test"]:
        source_input_path = os.path.join(config.data_dir, f"{split}.{config.source_lang}")
        target_input_path = os.path.join(config.data_dir, f"{split}.{config.target_lang}")
        output_path = os.path.join(config.cache_dir, f"{split}.tsv")
        create_translation_data(source_input_path, target_input_path, output_path)
        data_files[split] = [output_path]

    return data_files


if __name__ == "__main__":
    config = Config()

    data_files = create_wmt16_data_files(config)
    dataset_dict = load_dataset(
        'csv',
        delimiter='\t',
        column_names=[config.source_lang, config.target_lang],
        data_files=data_files
    )
    dataset_dict_tokenized = dataset_dict.map(
        batch_tokenize_fn,
        batched=True
    )

    model_name = config.model_checkpoint.split("/")[-1]
    output_dir = os.path.join(
        config.cache_dir,
        f"{model_name}_{config.source_lang}-{config.target_lang}"
    )
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=config.batch_size,
        predict_with_generate=True
    )
    data_collator = DataCollatorForSeq2Seq(config.tokenizer, model=config.model)
    rouge_score = evaluate.load("rouge", cache_dir=config.cache_dir)
    sacrebleu_score = evaluate.load("sacrebleu", cache_dir=config.cache_dir)
    trainer = Seq2SeqTrainer(
        config.model,
        args,
        train_dataset=dataset_dict_tokenized["train"],
        eval_dataset=dataset_dict_tokenized["val"],
        data_collator=data_collator,
        tokenizer=config.tokenizer,
        compute_metrics=compute_metrics,
    )
    print(trainer.evaluate())
