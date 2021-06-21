from .model import *
import json
import os
from collections import namedtuple
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

if __name__ == "__main__":
    config = open("config.json").read()
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
