import torch

from torch import nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

from callback.metrics_logger import _configure_logger
from metrics.metrics import RigidTransformationMetrics


class TestSequenceMetricsLogger(Callback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_sequences = False

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.sequences = getattr(trainer.datamodule, 'sequences', None)

        if self.sequences is not None:
            self.log_sequences = True
            self.logger = _configure_logger(self.log_path, 'test_sequences')
            self.test_metrics = nn.ModuleList([RigidTransformationMetrics(prefix='test').to(pl_module.device) for _ in range(len(self.sequences))])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.log_sequences:
            self.test_metrics[dataloader_idx].update(outputs['pred'][:,:3,:], batch['transform_gt'][:,:3,:])

    def on_test_end(self, trainer, pl_module):
        if self.log_sequences:
            for i in range(len(self.sequences)):
                metrics = self.test_metrics[i].compute()
                self.logger.info(format_dict2(metrics, self.sequences[i], 'test'))

def format_dict2(dict, sequence, stage):
    d = {k.replace(stage+'_', ''): (v.item() if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}
    return 'Sequence: %s, Rot_MSE: %f, Rot_RMSE: %f, Rot_MAE: %f, Rot_R2: %f, Trans_MSE: %f, Trans_RMSE: %f, Trans_MAE: %f, Trans_R2: %f' % (sequence, d['r_mse'], d['r_rmse'], d['r_mae'], d['r_r2score'], d['t_mse'], d['t_rmse'], d['t_mae'], d['t_r2score'])
