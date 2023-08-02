import os
import numpy as np

from lightning.pytorch import LightningModule
from lightning.pytorch import Callback
from util.file_helper import create_dir

class TransportLogger(Callback):
    def __init__(self, n_samples=2):
        self.n_samples = n_samples

    def on_validation_start(self, trainer, pl_module: LightningModule) -> None:
        self.log_path = os.path.join(trainer.log_dir, 'transport')
        self.transport_matrices = []
        if not os.path.exists(self.log_path):
            create_dir(self.log_path)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if 'transport' in outputs and outputs['transport'] is not None:
            if not len(self.transport_matrices) * outputs['transport'].size(0) > self.n_samples:
                self.transport_matrices.append(outputs['transport'].detach().cpu().numpy())

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            if len(self.transport_matrices) == 0:
                return

            matrices = np.concatenate(self.transport_matrices)
            save_n = min(self.n_samples, matrices.shape[0])

            with open(os.path.join(self.log_path, str(pl_module.current_epoch) + '.npy'), 'wb') as f:
                np.save(f, matrices[:save_n])
