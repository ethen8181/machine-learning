"""
This script trains a clip variant model illustrating how to implementing
cross GPU in batch negative training

nohup python3 clip/main.py > clip.log 2>&1 &
"""
import os
import torch
import utils
import models
import random
import numpy as np
import pytorch_lightning as pl
from time import perf_counter
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class Config:
    cache_dir: str = "./clip"
    base_dir: str = os.path.join(cache_dir, "flickr30k_images")
    image_dir: str = os.path.join(base_dir, "flickr30k_images")
    captions_path: str = os.path.join(base_dir, "results.csv")
    val_size: float = 0.1
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    num_gpus: int = 8

    lr: float = 0.0001
    weight_decay: float = 0.0005
    temperature: float = 1.0
    epochs: int = 4
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clip module
    image_encoder_model: str = 'resnet50'
    image_embedding_dim: int = 2048
    image_size = 224
    text_encoder_model: str = "distilbert-base-uncased"
    text_embedding_dim: int = 768
    text_max_length: int = 200
    # for both image encoder and text encoder
    cross_gpu_negatives: bool = True
    pretrained: bool = True 
    trainable: bool = True
    # projection head
    projection_dim = 256 
    dropout = 0.1
    
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


if __name__ == "__main__":
    config = Config()
    df_train, df_val = utils.create_train_val_df(config.captions_path, config.val_size)

    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
    data_loader_train = utils.build_data_loaders(config, df_train, tokenizer, mode="train")
    data_loader_val = utils.build_data_loaders(config, df_val, tokenizer, mode="val")
    clip = models.ClipModel(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.num_gpus,
        max_epochs=config.epochs,
        precision=16,
        enable_progress_bar=True,
        log_every_n_steps=500,
    )
    t1_start = perf_counter()
    trainer.fit(clip, data_loader_train, data_loader_val)
    t1_stop = perf_counter()
    print("Training elapsed time:", t1_stop - t1_start)
    print("saved checkpoint to:", trainer.checkpoint_callback.best_model_path)
