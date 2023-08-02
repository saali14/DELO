#!/usr/bin/env python
# -*- coding: utf-8 -*-


from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch3d.structures import Pointclouds

#from pytorch_lightning.core.lightning import LightningModule
from lightning.pytorch import LightningModule

from torch.optim.lr_scheduler import MultiStepLR

from metrics.mean import MeanValue
from metrics.metrics import RigidTransformationMetrics
from util.timer import Timer

from data.dataloader import transforms as Transforms
from pytorch3d.ops import points_alignment as pa
#from pytorch3d.ops import iterative_closest_point


class ICP(LightningModule):
    def __init__(self, max_iter=100,
                 rmse_thr=1e-6,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.max_iter = max_iter
        self.rmse_thr = rmse_thr

        self.metrics = RigidTransformationMetrics()
        self.val_loss = MeanValue()
        self.forward_time = Timer()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--rmse_thr', type=float, help='RMSE threshold to terminate.')
        parser.add_argument('--max_iter', type=int, help='Max iterations.')
        return parser

    @staticmethod
    def get_default_batch_transform(num_points=5000, **kwargs):
        return [Transforms.SetDeterministic(),
                Transforms.Resampler(num_points, upsampling=True),
                Transforms.ShufflePoints(),
                Transforms.AddBatchDimension()]

    def forward(self, src, tgt):
        icp_solution = pa.iterative_closest_point(src, tgt, max_iterations=self.max_iter, relative_rmse_thr=self.rmse_thr)
        info = dict()
        info['converged'] = icp_solution.converged
        return torch.cat((icp_solution.RTs[0].transpose(-1, -2), icp_solution.RTs[1].unsqueeze(2)), dim=2), info

    def training_step(self, batch, _):
        if isinstance(batch['points_src'], torch.Tensor):
            src = batch['points_src'][:, :, :3]
            tgt = batch['points_target'][:, :, :3]
        else:
            src = Pointclouds(batch['points_src'])
            tgt = Pointclouds(batch['points_target'])
        gt_transforms = batch['transform_gt']
        pred_transforms, _ = self(src, tgt)

        loss =  self.transformation_loss(pred_transforms, gt_transforms)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch['points_src'], torch.Tensor):
            src = batch['points_src'][:, :, :3]
            tgt = batch['points_target'][:, :, :3]
        else:
            src = Pointclouds(batch['points_src'])
            tgt = Pointclouds(batch['points_target'])
        pred_transforms, _ = self(src, tgt)
        gt_transforms = batch['transform_gt'][:, :3, :]

        loss = self.transformation_loss(pred_transforms, gt_transforms)
        metrics = self.metrics(pred_transforms, gt_transforms)
        self.val_loss.update(loss.detach().cpu())
        self.log('val_loss', loss, on_epoch=True)
        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch['points_src'], torch.Tensor):
            src = batch['points_src'][:, :, :3]
            tgt = batch['points_target'][:, :, :3]
        else:
            src = Pointclouds(batch['points_src'])
            tgt = Pointclouds(batch['points_target'])
        self.forward_time.tic()
        pred_transforms, _ = self(src, tgt)
        self.forward_time.toc()
        gt_transforms = batch['transform_gt'][:, :3, :]
        self.metrics.update(pred_transforms, gt_transforms)
        return {
            'pred': pred_transforms,
            # 'gt': gt_transforms,
            'dataloader_idx': dataloader_idx
        }

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        # TODO remove to device with lightning 1.3
        src = torch.tensor(batch['points_src'][:, :, :3], device=self.device)
        tgt = torch.tensor(batch['points_target'][:, :, :3], device=self.device)
        return self(src, tgt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_steps, gamma=0.1)
        return [optimizer], [scheduler]

    def transformation_loss(self, pred, gt):
        loss = nn.MSELoss(reduction='mean')
        identity = torch.eye(3, device=self.device).unsqueeze(0).repeat(pred.size(0), 1, 1)
        return loss(torch.matmul(pred[:, :, :3].transpose(2, 1), gt[:, :3, :3]),
                    identity) + loss(pred[:, :, 3], gt[:, :3, 3])
