import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_euler_angles


# code from: https://github.com/aamini/evidential-deep-learning
class DenseNormalGamma(nn.Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(DenseNormalGamma, self).__init__()
        self.n_in = n_input
        self.n_out = 4 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(x, self.n_tasks, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)

        return torch.stack([gamma, nu, alpha, beta]).to(x.device)

class RegistrationEvidence(nn.Module):
    def __init__(self, emb_dims):
        super(RegistrationEvidence, self).__init__()
        self.emb_dims = emb_dims
        #self.evidential = DenseNormalGamma((emb_dims // 4) + 6, n_out_tasks=6)
        self.evidential = DenseNormalGamma(48, n_out_tasks=6)
        #self.evidential = DenseNormalGamma(6, n_out_tasks=6)

        self.pose_regressor = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU(),
                                    nn.Linear(emb_dims // 8, emb_dims // 8)
                                )

        self.regressor = nn.Sequential(nn.Linear((emb_dims // 8) + 6, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.BatchNorm1d(64),
                                            nn.ReLU(),
                                            nn.Linear(64, 48)
                                            )

    def forward(self, src_embedding, tgt_embedding, rot, trans):
        #print(src_embedding.size())
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1).max(dim=-1)[0]
        #print(embedding.size())
        trans1 = self.pose_regressor(embedding)
        r_euler = matrix_to_euler_angles(rot, convention='XYZ')
        transf = torch.cat((trans1, r_euler, trans), dim=1)
        #transf = torch.cat(( r_euler, trans), dim=1)
        reg = self.regressor(transf)
        evidential = self.evidential(reg)

        #print(trans, evidential[..., 3:][0])
        #gamma, nu, alpha, beta = evidential
        #trans_aleatoric = beta[:, 3:] / (alpha[:, 3:] - 1)
        #print(trans_aleatoric / nu[:, 3:])

        return evidential
