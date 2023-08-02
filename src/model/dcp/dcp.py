#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy
import math
import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

#from pytorch_lightning.core.lightning import LightningModule
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import MultiStepLR

from loss.auxiliary import UnbalancedOptimalTransportLoss
from loss.registration import RegistrationLoss, RegistrationLossL1
from loss.total import TotalLoss

from metrics.mean import MeanValue
from metrics.metrics import RigidTransformationMetrics

from model.devilslam.reg_head import UOT_Head, OT_Head

from util.file_helper import create_dir
from util.pointcloud import nested_list_to_tensor
from util.timer import Timer

from data.dataloader import transforms as Transforms

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class MLPHead(nn.Module):
    def __init__(self, emb_dims):
        super(MLPHead, self).__init__()

        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation, None

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input

class Transformer(nn.Module):
    def __init__(self, emb_dims, n_blocks, dropout, ff_dims, n_heads):
        super(Transformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

class SVDHead(nn.Module):
    def __init__(self, emb_dims):
        super(SVDHead, self).__init__()
        self.emb_dims = emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)
            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3), None

class DCP(LightningModule):
    def __init__(self, lr=1e-4,
                 lr_steps=(15, 30, 45),
                 emb_dims=512,
                 emb_nn='dgcnn',
                 pointer='transformer',
                 n_blocks=1,
                 dropout=0.0,
                 ff_dims=1024,
                 n_heads=4,
                 head='svd',
                 reg_loss='mat',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_steps = lr_steps
        self.emb_dims = emb_dims
        self.emb_nn_type = emb_nn
        
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        elif emb_nn == 'sparse':
            from model.dslam.local_desc import SparseDesc
            self.emb_nn = SparseDesc()
        else:
            raise Exception('Not implemented')

        if pointer == 'identity':
            self.pointer = Identity()
        elif pointer == 'transformer':
            self.pointer = Transformer(emb_dims, n_blocks, dropout, ff_dims, n_heads)
        else:
            raise Exception("Not implemented")

        self.auxiliary_loss = None
        loss_weights = { 'reg_loss': 1.0, }

        if head == 'mlp':
            self.head = MLPHead(emb_dims)
        elif head == 'svd':
            self.head = SVDHead(emb_dims)
        elif head == 'uot':
            self.head = UOT_Head(device=self.device)
            self.auxiliary_loss = UnbalancedOptimalTransportLoss()
            loss_weights['aux_loss'] = 0.05
        elif head == 'ot':
            self.head = OT_Head(partial=False)
        elif head == 'partial':
            self.head = OT_Head(partial=True)
            self.auxiliary_loss = UnbalancedOptimalTransportLoss()
            loss_weights['aux_loss'] = 0.05
        else:
            raise Exception('Not implemented')

        self.metrics = RigidTransformationMetrics(prefix='val')
        self.val_loss = MeanValue()
        self.train_loss = MeanValue()
        self.train_metrics = RigidTransformationMetrics(prefix='train')
        self.loss = TotalLoss(weights=loss_weights)
        self.reg_loss = reg_loss

        if self.reg_loss == 'mat':
            self.registration_loss = RegistrationLoss(rot_weight=5.)
        elif self.reg_loss == 'l1':
            self.registration_loss = RegistrationLossL1()
        else:
            raise NotImplementedError
        self.forward_time = Timer()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--emb_nn', type=str, metavar='N', choices=['pointnet', 'dgcnn', 'sparse'], help='Embedding [pointnet, dgcnn]')
        parser.add_argument('--pointer', type=str, metavar='N', choices=['identity', 'transformer'], help='Attention-based pointer [identity, transformer]')
        parser.add_argument('--head', type=str, metavar='N', choices=['mlp', 'svd', 'uot', 'ot', 'partial'], help='Head to use, [mlp, svd]')
        parser.add_argument('--emb_dims', type=int, metavar='N', help='Dimension of embeddings')
        parser.add_argument('--n_blocks', type=int, metavar='N', help='Num of blocks of encoder&decoder')
        parser.add_argument('--n_heads', type=int, metavar='N', help='Num of heads in multiheadedattention')
        parser.add_argument('--ff_dims', type=int, metavar='N', help='Num of dimensions of fc in transformer')
        parser.add_argument('--dropout', type=float, metavar='N', help='Dropout ratio in transformer')
        parser.add_argument('--lr', type=float, help='Learning rate during training')
        parser.add_argument('--lr_steps', type=int, nargs='+', help='Steps to decrease lr')
        parser.add_argument('--reg_loss', type=str, choices=['mat', 'l1'])
        return parser

    @staticmethod
    def get_default_batch_transform(num_points=1024, **kwargs):
        return [Transforms.SetDeterministic(),
                Transforms.ShufflePoints(),
                Transforms.Resampler(num_points, upsampling=True),
                Transforms.AddBatchDimension()]

    def forward(self, src, tgt):
        info = {}
        if self.emb_nn_type == 'sparse':
            src_embedding, src_coord = self.emb_nn(src)
            tgt_embedding, tgt_coord = self.emb_nn(tgt)
            rotation_ab, translation_ab, transport = self.head(src_embedding.transpose(2, 1), tgt_embedding.transpose(2, 1), src_coord.transpose(2, 1), tgt_coord.transpose(2, 1))
            info['src'] = src_coord
            info['tgt'] = tgt_coord
        else:
            src_embedding = self.emb_nn(src)
            tgt_embedding = self.emb_nn(tgt)

            src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

            src_embedding = src_embedding + src_embedding_p
            tgt_embedding = tgt_embedding + tgt_embedding_p

            rotation_ab, translation_ab, transport = self.head(src_embedding, tgt_embedding, src, tgt)
        info['transport'] = transport
        return torch.cat((rotation_ab, translation_ab.unsqueeze(2)), dim=2), info

    def training_step(self, batch, batch_idx):
        if self.emb_nn_type == 'sparse':
            src = batch['points_src']
            tgt = batch['points_target']
            if self.reg_loss == 'l1':
                loss_src = nested_list_to_tensor(batch['points_src_copy'], num_points=5000, device=self.device)
        else:
            src = batch['points_src'][:, :, :3].transpose(2, 1)
            tgt = batch['points_target'][:, :, :3].transpose(2, 1)
            if self.reg_loss == 'l1':
                loss_src = batch['points_src'][:, :, :3]
        gt_transforms = batch['transform_gt'][:,:3,:]
        pred_transforms, info = self(src, tgt)

        if self.reg_loss == 'l1':
            losses = self.registration_loss(loss_src, pred_transforms, gt_transforms)
        else:
            losses = self.registration_loss(pred_transforms, gt_transforms)
        if self.auxiliary_loss is not None:
            if self.emb_nn_type == 'sparse':
                losses.update(self.auxiliary_loss(info['src'], info['tgt'], info['transport'], gt_transforms))
            else:
                losses.update(self.auxiliary_loss(batch['points_src'][:, :, :3], batch['points_target'][:, :, :3], info['transport'], gt_transforms))

        losses.update(self.loss(losses))
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_loss.update(losses['train_total_loss'].detach().cpu())  # save loss for callback logger

        metrics = self.train_metrics(pred_transforms, gt_transforms)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return losses['train_total_loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if self.emb_nn_type == 'sparse':
            src = batch['points_src']
            tgt = batch['points_target']
            if self.reg_loss == 'l1':
                loss_src = nested_list_to_tensor(batch['points_src_copy'], num_points=5000, device=self.device)
        else:
            src = batch['points_src'][:, :, :3].transpose(2, 1)
            tgt = batch['points_target'][:, :, :3].transpose(2, 1)
            if self.reg_loss == 'l1':
                loss_src = batch['points_src'][:, :, :3]
        pred_transforms, info = self(src, tgt)
        gt_transforms = batch['transform_gt'][:,:3,:]

        if self.reg_loss == 'l1':
            losses = self.registration_loss(loss_src, pred_transforms, gt_transforms, prefix='val')
        else:
            losses = self.registration_loss(pred_transforms, gt_transforms, prefix='val')
        if self.auxiliary_loss is not None:
            if self.emb_nn_type == 'sparse':
                losses.update(self.auxiliary_loss(info['src'], info['tgt'], info['transport'], gt_transforms, prefix='val'))
            else:
                losses.update(self.auxiliary_loss(batch['points_src'][:, :, :3], batch['points_target'][:, :, :3], info['transport'], gt_transforms, prefix='val'))

        losses.update(self.loss(losses, prefix='val'))
        self.log_dict(losses, on_epoch=True, sync_dist=True, on_step=True, prog_bar=True, logger=True)
        self.val_loss.update(losses['val_total_loss'].detach().cpu())  # save loss for callback logger

        metrics = self.metrics(pred_transforms, gt_transforms)
        self.log_dict(metrics, on_epoch=True, sync_dist=True, on_step=True, prog_bar=True, logger=True)

        #print(self.trainer.log_dir)

        if self.global_rank == 0 and info['transport'] is not None:
            if not batch['label'][0] in self.transport_matrices:
                self.transport_matrices[batch['label'][0]] = {
                    'transport': info['transport'][0].detach().cpu().numpy(),
                    'tgt': batch['target_id'][0].item(),
                    'src': batch['src_id'][0].item(),
                    'scene': batch['label'][0],
                    'src_points': batch['points_src'][0].detach().cpu().numpy(),
                    'tgt_points': batch['points_target'][0].detach().cpu().numpy(),
                }


            # print(batch['label'])
            # print(batch['src_id'])
            # if not len(self.transport_matrices) * info['transport'].size(0) > 2:
            #     self.transport_matrices.append(info['transport'].detach().cpu().numpy())
            #     mat = info['transport'].detach().cpu().numpy()
            #     print(mat.size)
            #     print(np.count_nonzero(mat > 1e-4))

        return {
            'pred': pred_transforms,
            # 'gt': gt_transforms,
            'dataloader_idx': dataloader_idx,
            #'transport': info['transport'],
        }

    def on_validation_start(self) -> None:
        if self.global_rank == 0:
            log_path = os.path.join(self.logger.log_dir, 'transport')
            self.transport_matrices = {}
            if not os.path.exists(log_path):
                create_dir(log_path)

    def on_validation_end(self) -> None:
        if self.global_rank == 0 and self.current_epoch % 5 == 0:

            for sample in self.transport_matrices.values():
                with open(os.path.join(self.logger.log_dir, 'transport', str(self.current_epoch) + '_' + str(sample['scene']) + '_' + str(sample['tgt']) + '_' + str(sample['src']) + '.npz'), 'wb') as f:
                    mat = sample['transport']
                    mat[mat < 1e-6] = 0
                    scipy.sparse.save_npz(f, scipy.sparse.csr_matrix(mat))
                #with open(os.path.join(self.logger.log_dir, 'transport', str(sample['scene']) + '_' + str(sample['src']) + '.npy'), 'wb') as f:
                #    np.save(f, sample['src_points'])
                #with open(os.path.join(self.logger.log_dir, 'transport', str(sample['scene']) + '_' + str(sample['tgt']) + '.npy'), 'wb') as f:
                #    np.save(f, sample['tgt_points'])

            #
            # if len(self.transport_matrices) == 0:
            #     return
            #
            # matrices = np.concatenate(self.transport_matrices)
            # save_n = min(1, matrices.shape[0])
            #
            # with open(os.path.join(self.trainer.log_dir, 'transport', str(self.current_epoch) + '.npy'), 'wb') as f:
            #     np.save(f, matrices[:save_n])
            #
            # with open(os.path.join(self.trainer.log_dir, 'transport', str(self.current_epoch) + '.npz'), 'wb') as f:
            #     mat = matrices[:save_n].squeeze()
            #     mat[mat < 1e-6] = 0
            #     scipy.sparse.save_npz(f, scipy.sparse.csr_matrix(mat))

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        src = batch['points_src'][:, :, :3].transpose(2, 1)
        tgt = batch['points_target'][:, :, :3].transpose(2, 1)
        self.forward_time.tic()
        pred_transforms, _ = self(src, tgt)
        self.forward_time.toc()
        gt_transforms = batch['transform_gt'][:,:3,:]

        self.metrics.update(pred_transforms, gt_transforms)

        return {
            'pred': pred_transforms,
            #'gt': gt_transforms,
            'dataloader_idx': dataloader_idx
        }

    def test_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics.compute(), on_epoch=True, sync_dist=True, on_step=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        # TODO remove to device with lightning 1.3
        src = torch.tensor(batch['points_src'][:, :, :3], device=self.device).transpose(2, 1)
        tgt = torch.tensor(batch['points_target'][:, :, :3], device=self.device).transpose(2, 1)
        return self(src, tgt)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_steps, gamma=0.1)
        return [optimizer], [scheduler]
