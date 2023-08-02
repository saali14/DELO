import math
import torch

from torch import nn

def cosine_distance(x1, x2, eps=1e-6):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def l2_distance(x1, x2, eps=1e-6):
    return torch.cdist(x1, x2, p=2.0).clamp(min=eps, max=10)

def unbalanced_optimal_transport(C, l, rho, iter=5, eps=1e-8):
    dim_a, dim_b = C.size()

    a = torch.ones(dim_a, dtype=torch.float32, device=C.device) / dim_a
    b = torch.ones(dim_b, dtype=torch.float32, device=C.device) / dim_b
    v = torch.ones(dim_b, dtype=torch.float32, device=C.device) / dim_b

    K = torch.exp(-C / l.clamp(min=eps))

    fi = rho / (rho + l).clamp(min=eps)

    for i in range(iter):
        u = (a / torch.matmul(K, v).clamp(min=eps)) ** fi
        v = (b / torch.matmul(K.T, u).clamp(min=eps)) ** fi

    return u * K * v.T

# method from from RPMNet (https://github.com/yewzijian/RPMNet/blob/master/src/models/rpmnet.py)
def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets
    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)
    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-6)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    det = torch.det(rot_mat_pos)[:, None, None]
    rot_mat = torch.where(det > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform

# method from from RPMNet (https://github.com/yewzijian/RPMNet/blob/master/src/models/rpmnet.py)
def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1
    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)
    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha

def simple_sinkhorn(MatrixA, n_iter = 20):
    #performing simple Sinkhorn iterations.

    for i in range(n_iter):
        MatrixA /= MatrixA.sum(dim=1, keepdim=True)
        MatrixA /= MatrixA.sum(dim=2, keepdim=True)
    return MatrixA

# following: https://github.com/PythonOT/POT/blob/master/ot/bregman.py
def optimal_transport(C, reg, n_iter=5, eps=1e-8):
    dim_a, dim_b = C.size()

    a = torch.ones(dim_a, dtype=torch.float32, device=C.device) / dim_a
    b = torch.ones(dim_b, dtype=torch.float32, device=C.device) / dim_b

    u = torch.ones(dim_a, dtype=torch.float32, device=C.device) / dim_a
    v = torch.ones(dim_b, dtype=torch.float32, device=C.device) / dim_b

    K = torch.exp(C / (-reg.clamp(min=eps)))
    Kp = (1 / a).reshape(-1, 1) * K

    for i in range(n_iter):
        KtransposeU = torch.matmul(K.T, u)
        v = b / KtransposeU.clamp(min=eps)
        u = 1. / torch.matmul(Kp, v).clamp(min=eps)

    return u.reshape((-1, 1)) * K * v.reshape((1, -1))

# following: https://github.com/PythonOT/POT/blob/master/ot/partial.py
def partial_optimal_transport(C, reg, n_iter=5, eps=1e-8):
    dim_a, dim_b = C.size()

    a = torch.ones(dim_a, dtype=torch.float32, device=C.device) / dim_a
    b = torch.ones(dim_b, dtype=torch.float32, device=C.device) / dim_b

    dx = torch.ones(dim_a, dtype=torch.float32, device=C.device)
    dy = torch.ones(dim_b, dtype=torch.float32, device=C.device)

    m = torch.min(torch.sum(a), torch.sum(b)) * 1.0

    K = torch.exp((-C) / reg.clamp(min=eps))
    K = K * (m / torch.sum(K).clamp(min=eps))


    for i in range(n_iter):
        K1 = torch.matmul(torch.diag(torch.minimum(a / torch.sum(K, dim=1).clamp(min=eps), dx)), K)
        K2 = torch.matmul(K1, torch.diag(torch.minimum(b / torch.sum(K1, dim=0).clamp(min=eps), dy)))
        K = K2 * (m / torch.sum(K2).clamp(min=eps))

    return K

class UOT_Head(nn.Module):
    def __init__(self, device):
        super(UOT_Head, self).__init__()

        self.l = nn.Parameter(torch.tensor(1., device=device))
        self.rho = nn.Parameter(torch.tensor(1., device=device))

    def forward(self, src_embedding, tgt_embedding, src, tgt):
        Ts = []
        uots = []
        for i in range(src.size(0)):
            C = cosine_distance(src_embedding[i].T, tgt_embedding[i].T)
            uot = unbalanced_optimal_transport(C, l=self.l, rho=self.rho)
            uots.append(uot)

            weighted_ref = uot @ tgt[i].T / (torch.sum(uot, dim=1, keepdim=True) + 1e-6)
            transform = compute_rigid_transform(src[i].T.unsqueeze(0), weighted_ref.unsqueeze(0), weights=torch.sum(uot, dim=1).unsqueeze(0))
            Ts.append(transform)

        Ts = torch.cat(Ts, dim=0)
        return Ts[:, :3, :3], Ts[:, :3, 3], torch.stack(uots)

class OT_Head(nn.Module):
    def __init__(self, partial=True, feat_distance='softmax'):
        super(OT_Head, self).__init__()
        self.partial = partial
        self.rho = nn.Parameter(torch.tensor(1.))
        self.feat_distance = feat_distance

    def forward(self, src_embedding, tgt_embedding, src, tgt):
        Ts = []
        match_mats = []
        for i in range(src.size(0)):
            if self.feat_distance == 'cosine':
                C = cosine_distance(src_embedding[i].T, tgt_embedding[i].T)
            elif self.feat_distance == 'l2':
                C = l2_distance(src_embedding[i].T, tgt_embedding[i].T)
            elif self.feat_distance == 'softmax':
                scores = torch.matmul(src_embedding[i].T, tgt_embedding[i]) / math.sqrt(src_embedding[i].size(0))
                #C = - torch.softmax(scores, dim=1)
                #C = torch.softmax(scores, dim=1)


                #scores_tmp = torch.zeros_like(scores)
                #valid_mask = torch.matmul(src[i, 0, :].unsqueeze(0).T, tgt[i, 0, :].unsqueeze(0)) > 0
                #scores_tmp[valid_mask] = scores[valid_mask]
                #scores = scores_tmp


                #src_size = min(torch.count_nonzero(src[i, 0, :]), 2048)
                #tgt_size = min(torch.count_nonzero(tgt[i, 0, :]), 2048)

                #print(valid_mask)

                #print(src_size, tgt_size)

                #scores = torch.matmul(src_embedding[i, :, :src_size].T, tgt_embedding[i, :, :tgt_size]) / math.sqrt(src_embedding[i].size(0))


                C = -torch.log_softmax(scores, dim=1)



                #print(C.min(), C.max())
            else:
                raise NotImplementedError

            #C = cosine_distance(torch.ones_like(src_embedding[i].T), torch.ones_like(tgt_embedding[i].T))

            if self.partial:
                matches = partial_optimal_transport(C, self.rho)
            else:
                matches = optimal_transport(C, self.rho)

            match_mats.append(matches)


            #print(src)

            #print(torch.softmax(scores, dim=1).max(), matches.max())
            #print(torch.softmax(scores, dim=1).min(), matches.min())

            #C_log = torch.log(C.clamp(min=1e-6))
            #matches = sinkhorn(C_log.unsqueeze(0), slack=self.slack)[0]
            #matches = torch.exp(matches)

            #print(C)

            weighted_ref = matches @ tgt[i].T / (torch.sum(matches, dim=1, keepdim=True) + 1e-6)

            transform = compute_rigid_transform(src[i].T.unsqueeze(0), weighted_ref.unsqueeze(0),
                                                weights=torch.sum(matches, dim=1).unsqueeze(0))
            Ts.append(transform)


        Ts = torch.cat(Ts, dim=0)
        return Ts[:, :3, :3], Ts[:, :3, 3], torch.stack(match_mats)
