#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import MultiStepLR
from argparse import ArgumentParser

from loss.auxiliary import UnbalancedOptimalTransportLoss
from loss.evidence import EvidenceLoss
from loss.global_desc import TripletLoss
from loss.registration import ScaledAdaptiveRegistrationLoss, RegistrationLoss, RegistrationLossL1, RegistrationLossMSE
from loss.total import TotalLoss

from metrics.evidence import EvidenceMetric
from metrics.global_desc import GlobalDescMetric
from metrics.mean import MeanValue
from metrics.metrics import RigidTransformationMetrics
from model.dcp.dcp import PointNet, DGCNN, Identity, Transformer, MLPHead, SVDHead
from model.delo.evidence import RegistrationEvidence
from model.delo.global_desc import GlobalDescNet
from model.delo.reg_head import UOT_Head, OT_Head

from util.pointcloud import nested_list_to_tensor
from util.timer import Timer

from data.dataloader import transforms as Transforms


class DELO(LightningModule):
    def __init__(self, lr=1e-3,
                 lr_steps=(15, 30, 45),
                 emb_dims=512,
                 emb_nn='dgcnn',
                 pointer='transformer',
                 n_blocks=1,
                 dropout=0.0,
                 ff_dims=1024,
                 n_heads=4,
                 head='uot',
                 head_feat_distance='softmax',
                 disable_aux_loss=False,
                 use_evidence=False,
                 use_descriptor=False,
                 train_only_descriptor=False,
                 reg_loss='mat',
                 desc_dim=256,
                 desc_margin=1.0,
                 desc_clusters=32,
                 desc_src='embedding',
                 desc_no_gating=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_steps = lr_steps
        self.emb_dims = emb_dims
        self.use_evidence = use_evidence
        self.train_only_descriptor = train_only_descriptor

        self.emb_nn_type = emb_nn
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if pointer == 'identity':
            self.pointer = Identity()
        elif pointer == 'transformer':
            self.pointer = Transformer(emb_dims, n_blocks, dropout, ff_dims, n_heads)
        else:
            raise Exception("Not implemented")

        loss_weights = {'reg_loss': 1.0,}
        self.auxiliary_loss = None
        if head == 'mlp':
            self.head = MLPHead(emb_dims)
        elif head == 'svd':
            self.head = SVDHead(emb_dims)
        elif head == 'uot':
            self.head = UOT_Head(device=self.device)
            if not disable_aux_loss:
                self.auxiliary_loss = UnbalancedOptimalTransportLoss()
                loss_weights['aux_loss'] = 0.05
        elif head == 'ot':
            self.head = OT_Head(partial=False, feat_distance=head_feat_distance)
        elif head == 'partial':
            self.head = OT_Head(partial=True, feat_distance=head_feat_distance)
            if not disable_aux_loss:
                self.auxiliary_loss = UnbalancedOptimalTransportLoss()
                loss_weights['aux_loss'] = 0.05
        else:
            raise Exception('Not implemented')

        self.metrics = RigidTransformationMetrics(prefix='val')
        self.val_loss = MeanValue()
        self.train_metrics = RigidTransformationMetrics(prefix='train')
        self.train_loss = MeanValue()
        self.forward_time = Timer()

        self.reg_loss = reg_loss
        if self.reg_loss == 'mat':
            self.registration_loss = RegistrationLoss(rot_weight=5.)
        elif self.reg_loss == 'l1':
            self.registration_loss = RegistrationLossL1()
        elif self.reg_loss == 'mse':
            self.registration_loss = RegistrationLossMSE()
        else:
            raise NotImplementedError

        if self.use_evidence:
            self.evidence_nn = RegistrationEvidence(emb_dims)
            self.evidence_loss = EvidenceLoss()
            self.evidence_train_metric = EvidenceMetric(prefix='train')
            self.evidence_val_metric = EvidenceMetric(prefix='val')
            loss_weights['evidence_loss'] = 0.0001

        if self.train_only_descriptor:
            loss_weights = {
                'triplet_loss': 1.0,
            }
            self.auxiliary_loss = None
            for param in self.emb_nn.parameters():
                param.requires_grad = False
            if self.desc_src == 'pointer':
                for param in self.pointer.parameters():
                    param.requires_grad = False
            if hasattr(self, 'evidence_nn'):
                for param in self.evidence_nn.parameters():
                    param.requires_grad = False

        self.loss = TotalLoss(weights=loss_weights)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--emb_nn', type=str, metavar='N', choices=['pointnet', 'dgcnn', 'sparse'], help='Backbones: [pointnet, dgcnn]')
        parser.add_argument('--pointer', type=str, metavar='N', choices=['identity', 'transformer'], help='Attention pointer: [identity, transformer]')
        parser.add_argument('--head', type=str, metavar='N', choices=['mlp', 'svd', 'uot', 'ot', 'partial'], help='Head to use, [mlp, svd]')
        parser.add_argument('--head_feat_distance', type=str, choices=['softmax', 'cosine', 'l2'])
        parser.add_argument('--emb_dims', type=int, metavar='N', help='Dimension of embeddings')
        parser.add_argument('--n_blocks', type=int, metavar='N', help='Num of blocks of encoder&decoder')
        parser.add_argument('--n_heads', type=int, metavar='N', help='Num of heads in multiheadedattention')
        parser.add_argument('--ff_dims', type=int, metavar='N', help='Num of dimensions of fc in transformer')
        parser.add_argument('--dropout', type=float, metavar='N', help='Dropout ratio in transformer')
        parser.add_argument('--lr', type=float, help='Learning rate during training')
        parser.add_argument('--lr_steps', type=int, nargs='+', help='Steps to decrease lr')
        parser.add_argument('--use_evidence', action='store_true', help='Predict prediction evidence.')
        parser.add_argument('--use_descriptor', action='store_true', help='Predict global descriptor.')
        parser.add_argument('--train_only_descriptor', action='store_true', help='Predict global descriptor.')
        parser.add_argument('--reg_loss', type=str, choices=['mat', 'l1', 'mse'], help='norm of transformation Loss')
        parser.add_argument('--desc_dim', type=int)
        parser.add_argument('--desc_margin', type=float)
        parser.add_argument('--desc_clusters', type=int)
        parser.add_argument('--desc_src', type=str, choices=['embedding', 'pointer'])
        parser.add_argument('--desc_no_gating', action='store_true', help='Disable context gating.')
        parser.add_argument('--disable_aux_loss', action='store_true', help='Disable auxiliary loss.')
        return parser

    @staticmethod
    def get_default_batch_transform(num_points=1024, voxel=False, **kwargs):
        if voxel:
            return [Transforms.SetDeterministic(),
                    Transforms.ShufflePoints(),
                    Transforms.VoxelCentroidResampler(num_points),
                    Transforms.Resampler(num_points, upsampling=True),
                    Transforms.AddBatchDimension()]
        return [Transforms.SetDeterministic(),
                Transforms.ShufflePoints(),
                Transforms.Resampler(num_points, upsampling=True),
                Transforms.AddBatchDimension()]

    def forward(self, src, tgt, neg=None):
        info = dict()
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        if not self.train_only_descriptor or self.desc_src == 'pointer':
            src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

            src_embedding_t = src_embedding + src_embedding_p
            tgt_embedding_t = tgt_embedding + tgt_embedding_p

            if not self.train_only_descriptor:
                rotation_ab, translation_ab, transport = self.head(src_embedding_t, tgt_embedding_t, src, tgt)
                info['transport'] = transport
                pred_transforms = torch.cat((rotation_ab, translation_ab.unsqueeze(2)), dim=2)
            else:
                pred_transforms = None
            info['src_embedding'] = src_embedding
            info['tgt_embedding'] = tgt_embedding
            info['src_embedding_t'] = src_embedding_t
            info['tgt_embedding_t'] = tgt_embedding_t
        else:
            pred_transforms = None
        if self.use_evidence and not self.train_only_descriptor:
            evidence = self.evidence_nn(src_embedding_t, tgt_embedding_t, rotation_ab, translation_ab)
            info['evidence'] = evidence

        return pred_transforms, info

    def training_step(self, batch, _):
        if self.emb_nn_type == 'sparse':
            src = batch['points_src']
            tgt = batch['points_target']
            if self.reg_loss == 'l1' or self.reg_loss == 'mse':
                loss_src = nested_list_to_tensor(batch['points_src_copy'], num_points=5000, device=self.device)
        else:
            src = batch['points_src'][:, :, :3].transpose(2, 1)
            tgt = batch['points_target'][:, :, :3].transpose(2, 1)
            if self.reg_loss == 'l1' or self.reg_loss == 'mse':
                loss_src = batch['points_src'][:, :, :3]
        
        neg = None
        if not self.train_only_descriptor:
            gt_transforms = batch['transform_gt'][:,:3,:]
        pred_transforms, info = self(src, tgt, neg)

        if self.train_only_descriptor:
            losses = {}
        elif self.reg_loss == 'l1' or self.reg_loss == 'mse':
            losses = self.registration_loss(loss_src, pred_transforms, gt_transforms)
        else:
            losses = self.registration_loss(pred_transforms, gt_transforms)
        
        
        if self.use_evidence and not self.train_only_descriptor:
            evidence_metric = self.evidence_train_metric(info['evidence'])
            self.log_dict(evidence_metric)
            losses.update(self.evidence_loss(gt_transforms, info['evidence']))
        if self.auxiliary_loss is not None:
            losses.update(self.auxiliary_loss(batch['points_src'][:, :, :3], batch['points_target'][:, :, :3], info['transport'], gt_transforms))
        if not self.train_only_descriptor:
            metrics = self.train_metrics(pred_transforms, gt_transforms)
            self.log_dict(metrics)
        
        losses.update(self.loss(losses))
        self.log_dict(losses)

        return losses['train_total_loss']

    def validation_step(self, batch, batch_idx):
        if self.emb_nn_type == 'sparse':
            src = batch['points_src']
            tgt = batch['points_target']
            if self.reg_loss == 'l1' or self.reg_loss == 'mse':
                loss_src = nested_list_to_tensor(batch['points_src_copy'], num_points=5000, device=self.device)
        else:
            src = batch['points_src'][:, :, :3].transpose(2, 1)
            tgt = batch['points_target'][:, :, :3].transpose(2, 1)
            if self.reg_loss == 'l1' or self.reg_loss == 'mse':
                loss_src = batch['points_src'][:, :, :3]
        
        neg = None
        pred_transforms, info = self(src, tgt, neg)
        
        if not self.train_only_descriptor:
            gt_transforms = batch['transform_gt'][:,:3,:]
        if self.train_only_descriptor:
            losses = {}
        elif self.reg_loss == 'l1' or self.reg_loss == 'mse':
            losses = self.registration_loss(loss_src, pred_transforms, gt_transforms)
        else:
            losses = self.registration_loss(pred_transforms, gt_transforms)
        
        if self.auxiliary_loss is not None:
            losses.update(self.auxiliary_loss(batch['points_src'][:, :, :3], batch['points_target'][:, :, :3], info['transport'], gt_transforms, prefix='val'))
        if self.use_evidence and not self.train_only_descriptor:
            evidence_metric = self.evidence_val_metric(info['evidence'])
            self.log_dict(evidence_metric, on_epoch=True, sync_dist=True)
            losses.update(self.evidence_loss(gt_transforms, info['evidence'], prefix='val'))

        losses.update(self.loss(losses, prefix='val'))
        self.log_dict(losses, on_epoch=True, sync_dist=True)

        if not self.train_only_descriptor:
            metrics = self.metrics(pred_transforms, gt_transforms)
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        src = batch['points_src'][:, :, :3].transpose(2, 1)
        tgt = batch['points_target'][:, :, :3].transpose(2, 1)
        self.forward_time.tic()
        pred_transforms, info = self(src, tgt)
        self.forward_time.toc()
        gt_transforms = batch['transform_gt'][:,:3,:]

        self.metrics.update(pred_transforms, gt_transforms)
        output = {
            'pred': pred_transforms,
            #'gt': gt_transforms,
            'dataloader_idx': dataloader_idx,
        }
        if 'src_global_desc' in info:
            output['src_global_desc'] = info['src_global_desc']
        if 'evidence' in info:
            output['evidence'] = info['evidence']
        return output

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics.compute())

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        # TODO remove to device with lightning 1.3
        src = torch.tensor(batch['points_src'][:, :, :3], device=self.device).transpose(2, 1)
        tgt = torch.tensor(batch['points_target'][:, :, :3], device=self.device).transpose(2, 1)
        return self(src, tgt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_steps, gamma=0.1)
        return [optimizer], [scheduler]
