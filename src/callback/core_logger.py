from typing import Optional

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from util.lightning import configure_logger


class CoreLogger(Callback):
    @rank_zero_only
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        configure_logger(trainer.log_dir)
