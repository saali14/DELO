#!/usr/bin/env python
# -*- coding: utf-8 -*-


from argparse import ArgumentParser

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import MultiStepLR

from metrics.mean import MeanValue
from metrics.metrics import RigidTransformationMetrics
from util.timer import Timer

from data.dataloader import transforms as Transforms


class Identity(LightningModule):
    def __init__(self, lr=1e-4,
                 lr_steps=(15, 30, 45),
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_steps = lr_steps

        self.metrics = RigidTransformationMetrics()
        self.val_loss = MeanValue()
        self.forward_time = Timer()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, help='Learning rate during training')
        parser.add_argument('--lr_steps', type=int,
                            nargs='+', help='Steps to decrease lr')
        return parser

    @staticmethod
    def get_default_batch_transform(**kwargs):
        return [Transforms.SetDeterministic(),
                Transforms.ShufflePoints()]

    def forward(self, src):
        return torch.eye(4, device=self.device).unsqueeze(0).repeat((src.size(0), 1, 1))

    def training_step(self, batch, _):
        src = batch['points_src'][:, :, :3]
        gt_transforms = batch['transform_gt']
        pred_transforms = self(src)

        loss =  self.transformation_loss(pred_transforms, gt_transforms)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src = batch['points_src'][:, :, :3]
        pred_transforms = self(src)
        gt_transforms = batch['transform_gt']

        loss = self.transformation_loss(pred_transforms, gt_transforms)
        metrics = self.metrics(pred_transforms, gt_transforms)
        self.val_loss.update(loss.detach().cpu())
        self.log('val_loss', loss, on_epoch=True)
        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        src = batch['points_src'][:, :, :3]
        self.forward_time.tic()
        pred_transforms = self(src)
        self.forward_time.toc()
        gt_transforms = batch['transform_gt']
        self.metrics.update(pred_transforms, gt_transforms)
        return pred_transforms

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        # TODO remove to device with lightning 1.3
        src = torch.tensor(batch['points_src'][:, :, :3], device=self.device)
        return self(src), None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_steps, gamma=0.1)
        return [optimizer], [scheduler]

    def transformation_loss(self, pred, gt):
        loss = nn.MSELoss(reduction='mean')
        identity = torch.eye(3, device=self.device).unsqueeze(0).repeat(pred.size(0), 1, 1)
        return loss(torch.matmul(pred[:, :, :3].transpose(2, 1), gt[:, :3, :3]),
                    identity) + loss(pred[:, :, 3], gt[:, :3, 3])
