import os
import lightning.pytorch as pl

from typing import Optional


class SavePeriodicCheckpoint(pl.Callback):
    def __init__(self, dirpath, every_n=5, save_first=True, save_only_weights=True):
        super().__init__()
        self.dirpath = dirpath
        self.every_n = every_n
        self.save_first = save_first
        self.save_only_weights = save_only_weights

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, unused: Optional = None) -> None:
        if (self.save_first and trainer.current_epoch == 0) or (trainer.current_epoch + 1) % self.every_n == 0:
            path = os.path.join(self.dirpath, 'epoch_%d.ckpt' % trainer.current_epoch)
            trainer.save_checkpoint(filepath=path, weights_only=self.save_only_weights)
