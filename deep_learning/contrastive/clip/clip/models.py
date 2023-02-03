import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):

    def __init__(
        self, model_name, pretrained, trainable, embedding_dim, projection_dim, dropout
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.projection = ProjectionHead(embedding_dim, projection_dim, dropout)

    def forward(self, img_arr):
        image_feature = self.model(img_arr)
        image_embedding = self.projection(image_feature)
        return image_embedding


class TextEncoder(nn.Module):

    def __init__(
        self, model_name, pretrained, trainable, embedding_dim, projection_dim, dropout
    ):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.projection = ProjectionHead(embedding_dim, projection_dim, dropout)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # we are using CLS token hidden representation as sentence's embedding
        text_feature = output.last_hidden_state[:, 0, :]
        text_embedding = self.projection(text_feature)
        return text_embedding


class ClipModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.image_encoder = ImageEncoder(
            model_name=config.image_encoder_model,
            pretrained=config.pretrained,
            trainable=config.trainable,
            embedding_dim=config.image_embedding_dim,
            projection_dim=config.projection_dim,
            dropout=config.dropout
        )
        self.text_encoder = TextEncoder(
            model_name=config.text_encoder_model,
            pretrained=config.pretrained,
            trainable=config.trainable,
            embedding_dim=config.text_embedding_dim,
            projection_dim=config.projection_dim,
            dropout=config.dropout
        )

        self.save_hyperparameters()

    def forward(self, batch):
        # Getting text and image embedding (with same dimension)
        text_embedding = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        image_embedding = self.image_encoder(batch["image"])
        return text_embedding, image_embedding

    def _compute_loss(self, text_embedding, image_embedding):
        """
        Cross GPU in batch negatives

        Say there are N GPUs and each GPU gets M input pairs.
        When calculating similarities (and thus the logit scores across text inputs for each image)
        a GPU only needs to hold M image features, which needs to be matmul'ed with NM text features,
        resulting in a M×NM matrix. This computation is distributed (i.e. sharded) across N GPUs,
        and overall we have calculated NM×NM similarities across the GPUs.
        The loss we use is symmetric and the same happens w.r.t. text inputs.

        References
        ----------
        - https://github.com/openai/CLIP/issues/29
        - https://github.com/openai/CLIP/issues/111
        - https://github.com/openai/CLIP/issues/132
        - https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py
        - https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
        """
        device = text_embedding.device
        batch_size = text_embedding.shape[0]

        if self.config.cross_gpu_negatives and self.config.num_gpus > 1:
            # unlike torch.distributed.nn.all_gather, lightning's all gather returns tensor
            # with an additional first dimension of size world_size, we collapse it so it
            # mimics the behaviour of torch.distributed.nn.all_gather + torch.cat
            all_text_embedding = self.all_gather(text_embedding).view(-1, text_embedding.shape[1])
            all_image_embedding = self.all_gather(image_embedding).view(-1, image_embedding.shape[1])
            # @ is equivalent to torch.mm
            text_logit = text_embedding @ all_image_embedding.T / self.config.temperature
            image_logit = image_embedding @ all_text_embedding.T / self.config.temperature
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = labels + batch_size * dist.get_rank()
        else:
            # @ is equivalent to torch.mm
            text_logit = text_embedding @ image_embedding.T / self.config.temperature
            image_logit = image_embedding @ text_embedding.T / self.config.temperature
            labels = torch.arange(batch_size, device=device, dtype=torch.long)

        text_loss = F.cross_entropy(text_logit, labels, reduction="none")
        image_loss = F.cross_entropy(image_logit, labels, reduction="none")
        loss = (image_loss + text_loss) / 2.0
        return loss.mean()

    def training_step(self, batch, batch_idx):
        text_embedding, image_embedding = self.forward(batch)
        loss = self._compute_loss(text_embedding, image_embedding)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        text_embedding, image_embedding = self.forward(batch)
        loss = self._compute_loss(text_embedding, image_embedding)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        return optimizer
