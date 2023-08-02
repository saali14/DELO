import torch
import torch.nn as nn

from pytorch3d.transforms import matrix_to_euler_angles
from util.pointcloud import transform

class RegistrationLoss(nn.Module):
    def __init__(self,
                 rot_weight=1.,
                 trans_weight=1.):
        super(RegistrationLoss, self).__init__()

        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def __call__(self, pred, gt, prefix='train'):
        loss = nn.MSELoss(reduction='mean')
        identity = torch.eye(3, device=pred.device).unsqueeze(0).repeat(pred.size(0), 1, 1)
        rot_loss = loss(torch.matmul(pred[:, :, :3].transpose(2, 1), gt[:, :3, :3]), identity)
        trans_loss = loss(pred[:, :, 3], gt[:, :3, 3])
        reg_loss = self.rot_weight * rot_loss + self.trans_weight * trans_loss

        losses = dict()
        losses[prefix + '_reg_loss'] = reg_loss
        losses[prefix + '_rot_loss'] = rot_loss
        losses[prefix + '_trans_loss'] = trans_loss

        return losses

class AdaptiveRegistrationLoss(nn.Module):
    def __init__(self, device):
        super(AdaptiveRegistrationLoss, self).__init__()

        self.rot_weight = nn.Parameter(torch.tensor(1., device=device))
        self.trans_weight = nn.Parameter(torch.tensor(1., device=device))

    def __call__(self, pred, gt, prefix='train'):
        loss = nn.MSELoss(reduction='mean')
        identity = torch.eye(3, device=self.device).unsqueeze(0).repeat(pred.size(0), 1, 1)

        rot_loss = loss(torch.matmul(pred[:, :, :3].transpose(2, 1), gt[:, :3, :3]), identity)
        trans_loss = loss(pred[:, :, 3], gt[:, :3, 3])

        reg_loss = torch.exp(-self.rot_weight) * rot_loss \
                   + torch.exp(-self.trans_weight) * trans_loss \
                   + self.rot_weight + self.trans_weight

        losses = dict()
        losses[prefix + '_reg_loss'] = reg_loss
        losses[prefix + '_rot_loss'] = rot_loss
        losses[prefix + '_trans_loss'] = trans_loss
        losses[prefix + '_rot_weight'] = self.rot_weight
        losses[prefix + '_trans_weight'] = self.trans_weight

        return losses

class ScaledAdaptiveRegistrationLoss(nn.Module):
    def __init__(self, device):
        super(ScaledAdaptiveRegistrationLoss, self).__init__()

        self.rot_weight = nn.Parameter(torch.tensor(1., device=device))
        self.trans_weight = nn.Parameter(torch.tensor(1., device=device))

    def __call__(self, pred, gt, prefix='train'):
        loss = nn.MSELoss(reduction='none')
        identity = torch.eye(3, device=pred.device).unsqueeze(0).repeat(pred.size(0), 1, 1)

        rot_scale = 1 + torch.exp(-torch.abs(torch.rad2deg(matrix_to_euler_angles(gt[:, :3, :3], convention='XYZ'))).mean(dim=1)**2)
        trans_scale = 1 + torch.exp(-torch.abs(gt[:, :3, 3]).mean(dim=1)**2)

        rot_loss = (loss(torch.matmul(pred[:, :, :3].transpose(2, 1), gt[:, :3, :3]), identity).mean(dim=(1,2)) * rot_scale).mean()
        trans_loss = (loss(pred[:, :, 3], gt[:, :3, 3]).mean(dim=1) * trans_scale).mean()

        reg_loss = torch.exp(-self.rot_weight) * rot_loss \
                   + torch.exp(-self.trans_weight) * trans_loss \
                   + self.rot_weight + self.trans_weight

        losses = dict()
        losses[prefix + '_reg_loss'] = reg_loss
        losses[prefix + '_rot_loss'] = rot_loss
        losses[prefix + '_trans_loss'] = trans_loss
        losses[prefix + '_rot_weight'] = self.rot_weight
        losses[prefix + '_trans_weight'] = self.trans_weight

        return losses

class RegistrationLossL1(nn.Module):
    def __init__(self):
        super(RegistrationLossL1, self).__init__()
        self.criterion = nn.L1Loss(reduction='mean')

    def __call__(self, points, pred, gt, prefix='train'):
        t_pc_pred = transform(points, pred)
        t_pc_gt = transform(points, gt)
        loss = self.criterion(t_pc_pred, t_pc_gt)
        losses = dict()
        losses[prefix + '_reg_loss'] = loss

        return losses

class RegistrationLossMSE(nn.Module):
    def __init__(self):
        super(RegistrationLossMSE, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def __call__(self, points, pred, gt, prefix='train'):
        t_pc_pred = transform(points, pred)
        t_pc_gt = transform(points, gt)

        loss = self.criterion(t_pc_pred, t_pc_gt)

        losses = dict()
        losses[prefix + '_reg_loss'] = loss

        return losses