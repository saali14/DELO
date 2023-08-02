import torch
import torch.nn as nn


class UnbalancedOptimalTransportLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(UnbalancedOptimalTransportLoss, self).__init__()

        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.eps = eps

    def __call__(self, source, target, transport, gt, prefix='train'):
        source_t = (torch.matmul(gt[:, :3, :3], source.transpose(2, 1)) + gt[:, :3, 3].unsqueeze(2)).transpose(2, 1)
        losses = dict()
        losses[prefix + '_aux_loss'] = self.criterion((transport @ target) / torch.sum(transport, dim=2, keepdim=True).clamp(min=self.eps), source_t)
        return losses
