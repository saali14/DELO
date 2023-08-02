import os
import torch
import numpy as np

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

from metrics.metrics import RigidTransformationMetrics

class LogTestSequence(Callback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_sequences = False
        self.gts = {}
        self.preds = {}
        self.evidence = {}

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.sequences = getattr(trainer.datamodule, 'sequences', None)
        if self.sequences is not None:
            self.log_sequences = True
            self.test_metrics = torch.nn.ModuleList([RigidTransformationMetrics(prefix='test').to(pl_module.device) for _ in range(len(self.sequences))])
            for i in range(len(self.sequences)):
                self.gts[i] = []
                self.preds[i] = []
                self.evidence[i] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.log_sequences:
            self.gts[dataloader_idx].extend(torch.unbind(batch['transform_gt'][:,:3,:].cpu()))
            self.preds[dataloader_idx].extend(torch.unbind(outputs['pred'][:,:3,:].cpu()))
            if 'evidence' in outputs:
                gamma, nu, alpha, beta = outputs['evidence']
                aleatoric = beta / (alpha - 1)
                epistemic = aleatoric / nu
                self.evidence[dataloader_idx].extend(torch.unbind(epistemic.cpu()))

    def on_test_end(self, trainer, pl_module):
        if self.log_sequences:
            for i in range(len(self.sequences)):
                preds = torch.stack(self.preds[i])
                gts = torch.stack(self.gts[i])
                save_predictions(preds.numpy(), os.path.join(self.log_path, '%s_pred.txt' % str(i)))
                save_predictions(gts.numpy(), os.path.join(self.log_path, '%s_gt.txt' % str(i)))

def save_predictions(preds, save_path):
    lines = []
    for i in range(preds.shape[0]):
        lines.append('0 0 0 0 0 0 0')
        for j in range(3):
            lines.append(' '.join([str(x) for x in list(preds[i, j, :])]))
        lines.append('0 0 0 1')

    with open(save_path, 'w') as file:
        file.write('\n'.join(lines))
