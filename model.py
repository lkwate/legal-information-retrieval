import pandas as pd
import numpy as np
import os
from transformers import BigBirdTokenizer, BigBirdModel
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
        inputs = self.tokenizer(row['text'], return_tensors='pt', max_length=self.config.max_length, padding='max_length', truncation=True)
        return (inputs.input_ids, inputs.attention_mask)
    
class LegalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__(self)
        self.config = config
        
    def prepare_data(self):
        print('data preparation ...')
        self.data = LegalDataset(self.config)
        print('data preparation completed ...')
    
    def setup(self, stage):
        train_length = int(self.config.train_val_split_factor * len(self.data))
        val_length = len(self.data) - train_length
        self.data_train, self.data_val = random_split(self.data, [train_length, val_length])
    
    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=self.config.num_cpus, batch_size=self.config.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=self.config.num_cpus, batch_size=self.config.batch_size)

class LegalBigBird(pl.LightningModule):
    def __init__(self, config):
        super(LegalBigBird, self).__init__()
        print('model initialization...')
        self.config = config
        self.encoder = BigBirdModel.from_pretrained(self.config.model)
        self.mlm = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
    
        self.__num_mask_token = int(self.config.mask_factor * self.config.max_length)
        print('model initialization completed.')
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss)
        return loss
    
    def compute_loss(self, batch):
        input_ids, attention_mask = batch[0].squeeze(1), batch[1].squeeze(1)
        last_token_index = attention_mask.sum(dim=-1).tolist()
        masked_token_index = [np.random.choice(range(index), self.__num_mask_token).tolist() for index in last_token_index]
        masked_token_index = torch.tensor(masked_token_index).type_as(input_ids)
        labels = input_ids.gather(-1, masked_token_index)
        for i in range(input_ids.shape[0]):
            input_ids[i, :][masked_token_index[i]] = self.config.mask_token_id
        
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        output = torch.stack([torch.index_select(output[i, :], 0, masked_token_index[i, :]) for i in range(input_ids.shape[0])], dim=0)
        output = self.mlm(output)
        
        loss = self.criterion(output.view(-1, self.config.vocab_size), labels.view(-1))
        return loss