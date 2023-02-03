import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader


def create_train_val_df(captions_path, val_size):
    df = pd.read_csv(
        captions_path,
        sep="|",
        skiprows=1,
        names=["image", "caption_number", "caption"]
    )
    # remove extra white space up front
    df['caption'] = df['caption'].str.lstrip()
    df['caption_number'] = df['caption_number'].str.lstrip()
    # one of the rows is corrupted
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."

    max_id = df.shape[0] // 5
    df["ids"] = [id_ for id_ in range(max_id) for _ in range(5)]

    image_ids = np.arange(0, max_id)
    val_ids = np.random.choice(
        image_ids,
        size=int(val_size * len(image_ids)),
        replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in val_ids]
    df_train = df[df["ids"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["ids"].isin(val_ids)].reset_index(drop=True)
    return df_train, df_val


class CLIPDataset(Dataset):
    """
    image_filenames and cpations must have the same length; so, if there are
    multiple captions for each image, the image_filenames must have repetitive
    file names 
    """

    def __init__(self, image_dir, image_filenames, image_size, captions, tokenizer, text_max_length):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.captions = captions
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.transforms = self.get_transforms(image_size)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.captions[idx],
            truncation=True,
            max_length=self.text_max_length
        )
        image = cv2.imread(f"{self.image_dir}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        encoded["image"] = image
        return encoded

    def __len__(self):
        return len(self.captions)

    def get_transforms(self, image_size):
        return A.Compose(
            [
                A.Resize(image_size, image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


class DataCollatorForClip:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        text_feature = {
            "input_ids": [feature["input_ids"] for feature in features],
            "attention_mask": [feature["attention_mask"] for feature in features]
        }
        batch = self.tokenizer.pad(
            text_feature,
            padding=True,
            return_tensors="pt"
        )
        # convert to channel first
        image_feature = [torch.FloatTensor(feature["image"]) for feature in features]
        image = torch.stack(image_feature).permute(0, 3, 1, 2)
        batch["image"] = image
        return batch


def build_data_loaders(config, df, tokenizer, mode):
    dataset = CLIPDataset(
        config.image_dir,
        list(df["image"].values),
        config.image_size,
        list(df["caption"].values),
        tokenizer,
        config.text_max_length,
    )
    data_collator = DataCollatorForClip(tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=True if mode == "train" else False
    )
    return data_loader
