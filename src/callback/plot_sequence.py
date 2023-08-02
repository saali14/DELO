import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from pytorch3d.transforms import matrix_to_euler_angles
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from metrics.metrics import RigidTransformationMetrics


class PlotTestSequence(Callback):
    def __init__(self, log_path, plot_prediction=True):
        self.log_path = log_path
        self.log_sequences = False
        self.plot_prediction = plot_prediction

        self.gts = {}
        self.preds = {}
        if self.plot_prediction:
            self.poses = {}
        self.evidence = {}

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.sequences = getattr(trainer.datamodule, 'sequences', None)

        if self.sequences is not None:
            self.log_sequences = True
            self.test_metrics = nn.ModuleList([RigidTransformationMetrics(prefix='test').to(pl_module.device) for _ in range(len(self.sequences))])
            for i in range(len(self.sequences)):
                self.gts[i] = []
                self.preds[i] = []
                if self.plot_prediction:
                    self.poses[i] = []
                self.evidence[i] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.log_sequences:
            self.gts[dataloader_idx].extend(torch.unbind(batch['transform_gt'][:,:3,:].cpu()))
            self.preds[dataloader_idx].extend(torch.unbind(outputs['pred'][:,:3,:].cpu()))
            if self.plot_prediction:
                self.poses[dataloader_idx].extend(torch.unbind(batch['pose'][:, :3, :].cpu()))
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

                t_err_mean, t_err = trans_error(preds, gts)
                r_err_mean, r_err = rot_error(preds, gts)

                plot_error_hist(r_err.numpy(), os.path.join(self.log_path, self.sequences[i] + '_rot_error_hist.png'), type='rot')
                plot_error_hist(t_err.numpy(), os.path.join(self.log_path, self.sequences[i] + '_trans_error_hist.png'), type='trans')

                if self.plot_prediction:
                    gt_poses = torch.stack(self.poses[i])
                    gt_path = gt_poses[:, :3, 3].numpy()
                    plot_path(gt_path, t_err_mean.numpy(),
                              os.path.join(self.log_path, self.sequences[i] + '_trans_error.png'))
                    plot_path(gt_path, r_err_mean.numpy(),
                              os.path.join(self.log_path, self.sequences[i] + '_rot_error.png'))
                    plot_prediction(self.preds[i], gt_path, os.path.join(self.log_path, self.sequences[i] + '_prediction.png'))

                if len(self.evidence[i]) > 0:
                    plot_evidence(np.concatenate([r_err.numpy(), t_err.numpy()], axis=1), torch.stack(self.evidence[i]).numpy(), os.path.join(self.log_path, self.sequences[i] + '_evidence_%s'), plot=self.plot_prediction)

                save_predictions(preds.numpy(), os.path.join(self.log_path, '%s_pred.txt' % str(i)))
                save_predictions(gts.numpy(), os.path.join(self.log_path, '%s_gt.txt' % str(i)))

def trans_error(pred, gt):
    return torch.abs(pred[:, :3, 3] - gt[:, :3, 3]).mean(dim=1), pred[:, :3, 3] - gt[:, :3, 3]

def rot_error(pred, gt):
    r_gt_euler_deg = torch.rad2deg(matrix_to_euler_angles(gt[:, :3, :3], convention='XYZ'))
    r_pred_euler_deg = torch.rad2deg(matrix_to_euler_angles(pred[:, :3, :3], convention='XYZ'))

    return torch.abs(r_pred_euler_deg - r_gt_euler_deg).mean(dim=1), r_pred_euler_deg - r_gt_euler_deg

def plot_path(path, error, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(15, 15))

    df = pd.DataFrame({'x': path[:, 0], 'y': path[:, 2], 'error': error})
    df = df.sort_values(by=['error'])

    ax = sns.scatterplot(data=df, x='x', y='y', hue='error', palette='Reds', s=50, alpha=1)
    # ax = sns.lineplot(x=gt_path[:, 0], y=gt_path[:, 2], sort=False, hue=t_err, palette='viridis')

    norm = plt.Normalize(error.min(), error.max())
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    plt.savefig(save_path)
    plt.clf()

def plot_prediction(preds, gt_path, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid')

    curr_pose = np.eye(4)
    poses = [curr_pose]

    for pred in preds:
        t = np.eye(4)
        t[:3, :] = pred[:3, :].numpy()
        curr_pose = curr_pose @ t
        poses.append(curr_pose)

    path = np.stack(poses)[:, :3, 3]

    plt.figure(figsize=(15, 15))

    sns.lineplot(x=path[:, 0], y=path[:, 1], sort=False, color='red')
    sns.lineplot(x=gt_path[:, 2], y=-gt_path[:, 0], sort=False, color='blue')

    plt.savefig(save_path)
    plt.clf()

def plot_error_hist(errors, save_path, type='rot'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid')

    size = errors.shape[0]
    dim = np.concatenate([np.full(size, 0), np.full(size, 1), np.full(size, 2)])

    df = pd.DataFrame.from_dict({'x': errors.flatten(order='F'), 'dim': dim})

    plt.figure(figsize=(15, 15))

    if type == 'rot':
        sns.histplot(data=df, x='x', hue='dim', binwidth=0.01, binrange=(-2, 2))
    else:
        sns.histplot(data=df, x='x', hue='dim', binwidth=0.005, binrange=(-0.5, 0.5))

    plt.savefig(save_path)
    plt.clf()

def plot_evidence(errors, evidence, save_path, plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid')

    for i in range(6):

        df = pd.DataFrame.from_dict({'x': errors[:, i], 'y': evidence[:, i]})
        df.to_csv(save_path % str(i) + '.csv')

        if plot:
            fig = plt.figure(figsize=(15, 15))

            sns.histplot(data=df, x='x', y='y') #,  binwidth=0.01)

            plt.savefig(save_path % str(i) + '.png')
            plt.clf()
            plt.close(fig)

def save_predictions(preds, save_path):
    lines = []
    for i in range(preds.shape[0]):
        lines.append('0 0 0 0 0 0 0')
        for j in range(3):
            lines.append(' '.join([str(x) for x in list(preds[i, j, :])]))
        lines.append('0 0 0 1')

    with open(save_path, 'w') as file:
        file.write('\n'.join(lines))
