import logging
import torch
import os
import numpy as np

from pytorch3d.transforms import matrix_to_euler_angles
from lightning.pytorch.callbacks import Callback

class RecallCallback(Callback):
    def __init__(self,
                 out_path,
                 rot_thresh=15,
                 trans_thresh=0.3):
        self.logger = logging.getLogger('test_logger')
        self.rot_errors = []
        self.trans_errors = []
        self.out_path = out_path
        self.rot_thresh = rot_thresh
        self.trans_thresh = trans_thresh

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        target = batch['transform_gt']
        preds = outputs['pred'].detach()#.cpu()

        r_gt_euler_deg = torch.rad2deg(matrix_to_euler_angles(target[..., :3, :3], convention='XYZ'))
        r_pred_euler_deg = torch.rad2deg(matrix_to_euler_angles(preds[..., :3, :3], convention='XYZ'))
        t_gt = target[..., :3, 3]
        t_pred = preds[..., :3, 3]

        rot_error = torch.mean(torch.abs(r_pred_euler_deg - r_gt_euler_deg), dim=1)
        trans_error = torch.mean(torch.abs(t_pred - t_gt), dim=1)

        self.rot_errors += rot_error.tolist()
        self.trans_errors += trans_error.tolist()

    def on_test_end(self, trainer, pl_module):
        # import here to prevent open3d load ply error
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style='whitegrid')

        rot_errors = np.array(self.rot_errors)
        trans_errors = np.array(self.trans_errors)
        samples = rot_errors.shape[0]

        rot_recall = np.sum(rot_errors <= self.rot_thresh) / samples
        trans_recall = np.sum(trans_errors <= self.trans_thresh) / samples
        total = np.sum((rot_errors <= self.rot_thresh) * (trans_errors <= self.trans_thresh)) / samples
        self.logger.info('Rotation Recall: {}, Translation Recall: {}, Total: {}'.format(rot_recall, trans_recall, total))

        ax = sns.histplot(data=rot_errors, bins=100, cumulative=True, element='poly', fill=False, stat='probability')
        ax.set_xlim(0, 30)
        ax.set_xlabel('Rotation (deg)')
        ax.set_ylabel('Recall')
        plt.axvline(self.rot_thresh, linestyle='--', color='grey')
        plt.savefig(os.path.join(self.out_path, 'rot_recall.png'))
        plt.clf()

        ax2 = sns.histplot(data=trans_errors, bins=100, cumulative=True, element='poly', fill=False, stat='probability')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Translation (m)')
        ax2.set_ylabel('Recall')
        plt.axvline(self.trans_thresh, linestyle='--', color='grey')
        plt.savefig(os.path.join(self.out_path, 'trans_recall.png'))
        plt.clf()
