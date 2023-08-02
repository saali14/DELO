import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_euler_angles


class EvidenceLoss(nn.Module):
    def __init__(self, lmbda=2e-1):
        super(EvidenceLoss, self).__init__()
        self.lmbda = lmbda

    def __call__(self, gt, evidence, prefix='train'):
        r_euler = matrix_to_euler_angles(gt[:, :3, :3], convention='XYZ')
        trans = torch.cat((r_euler, gt[:, :3, 3]), dim=1)

        losses = dict()
        loss_nll_r, loss_reg_r = EvidentialRegression(r_euler, evidence[..., :3], lmbda=self.lmbda)
        losses[prefix + '_evidence_nll_loss_r'] = loss_nll_r
        losses[prefix + '_evidence_reg_loss_r'] = loss_reg_r

        loss_nll_t, loss_reg_t = EvidentialRegression(gt[:, :3, 3], evidence[..., 3:], lmbda=self.lmbda*0.1)
        #print(gt[:, :3, 3], evidence[..., 3:][0])
        losses[prefix + '_evidence_nll_loss_t'] = loss_nll_t
        losses[prefix + '_evidence_reg_loss_t'] = loss_reg_t

        loss_nll = loss_nll_r + loss_nll_t
        loss_reg = loss_reg_r + loss_reg_t

        losses[prefix + '_evidence_nll_loss'] = loss_nll
        losses[prefix + '_evidence_reg_loss'] = loss_reg
        losses[prefix + '_evidence_loss'] = loss_nll + loss_reg
        return losses


# code from: https://github.com/aamini/evidential-deep-learning
def reduce(val, reduction):
    if reduction == 'mean':
        val = val.mean()
    elif reduction == 'sum':
        val = val.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f"Invalid reduction argument: {reduction}")
    return val


def NIG_NLL(y: torch.Tensor,
            gamma: torch.Tensor,
            nu: torch.Tensor,
            alpha: torch.Tensor,
            beta: torch.Tensor, reduction='mean'):
    inter = 2 * beta * (1 + nu)

    nll = 0.5 * (np.pi / nu).log() \
          - alpha * inter.log() \
          + (alpha + 0.5) * (nu * (y - gamma) ** 2 + inter).log() \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)
    return reduce(nll, reduction=reduction)


def NIG_Reg(y, gamma, nu, alpha, reduction='mean'):
    error = (y - gamma).abs()
    evidence = 2. * nu + alpha
    return reduce(error * evidence, reduction=reduction)


def EvidentialRegression(y: torch.Tensor, evidential_output: torch.Tensor, lmbda=1.):
    gamma, nu, alpha, beta = evidential_output
    loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
    loss_reg = NIG_Reg(y, gamma, nu, alpha)
    return loss_nll, lmbda * loss_reg