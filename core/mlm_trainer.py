import json
import pandas as pd
import numpy as np
import os
from transformers import (
    BigBirdTokenizer,
    BigBirdForMaskedLM,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from collections import namedtuple


class LegalDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(config.dataset)
        self.tokenizer = BigBirdTokenizer.from_pretrained(config.tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]
        inputs = self.tokenizer(
            row["text"],
            return_tensors="pt",
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
        )
        return inputs


class LegalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__(self)
        self.config = config
        self.prepare_data()
        self.mlm_data_collator = DataCollatorForLanguageModeling(
            self.data.tokenizer, mlm_probability=self.config.mlm_probability
        )

    def prepare_data(self):
        print("data preparation ...")
        self.data = LegalDataset(self.config)
        train_length = int(self.config.train_val_split_factor * len(self.data))
        val_length = len(self.data) - train_length
        self.data_train, self.data_val = random_split(
            self.data, [train_length, val_length]
        )
        print("data preparation completed ...")

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            num_workers=self.config.num_cpus,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.mlm_data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            num_workers=self.config.num_cpus,
            batch_size=self.config.batch_size,
            collate_fn=self.mlm_data_collator,
        )


class LegalBigBird(pl.LightningModule):
    def __init__(self, config):
        super(LegalBigBird, self).__init__()
        print("model initialization...")
        self.config = config
        self.model = BigBirdForMaskedLM.from_pretrained(self.config.model)
        print("model initialization completed.")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.config.lr_decay, patience=self.config.patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return optimizer

    def _compute_loss(self, batch):
        batch = {k: v.squeeze(1) for k, v in batch.items()}
        loss = self.model(**batch).loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss)
        return loss


if __name__ == "__main__":
    config = open("config-mlm.json").read()
    config = json.loads(config)
    config = namedtuple("config", config.keys())(*config.values())

    model = LegalBigBird(config)
    dm = LegalDataModule(config)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", strict=False, verbose=True
    )

    trainer = pl.Trainer(
        gpus=-1,
        callbacks=[early_stop_callback],
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    trainer.fit(model, dm)
