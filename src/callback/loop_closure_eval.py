import os
import torch
import faiss
import numpy as np
import pandas as pd

from pytorch3d.transforms import matrix_to_euler_angles
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule
from callback.metrics_logger import _configure_logger
from metrics.metrics import RigidTransformationMetrics

class LoopClosureEvaluation(Callback):
    def __init__(self, log_path, plot_prediction=True, pos_threshold=0.02, max_pos_distance=4):
        self.log_path = log_path
        self.log_sequences = False
        self.plot_prediction = plot_prediction
        self.pos_threshold = pos_threshold
        self.max_pos_distance = max_pos_distance
        self.global_desc = {}
        self.poses = {}

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.sequences = getattr(trainer.datamodule, 'sequences', None)

        if self.sequences is not None:
            self.log_sequences = True
            self.logger = _configure_logger(self.log_path, 'loop_closure')
            for i in range(len(self.sequences)):
                self.global_desc[i] = []
                self.poses[i] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.log_sequences:
            self.global_desc[dataloader_idx].extend(torch.unbind(outputs['src_global_desc'].cpu()))
            self.poses[dataloader_idx].extend(torch.unbind(batch['pose'][:, :3, :].cpu()))

    def on_test_end(self, trainer, pl_module):
        if self.log_sequences:
            for i in range(len(self.sequences)):
                gt_poses = torch.stack(self.poses[i])
                gt_path = gt_poses[:, :3, 3].numpy()
                pdist = (gt_path.reshape(1, -1, 3) - gt_path.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))
                global_descs = self.global_desc[i]

                #res = faiss.StandardGpuResources()
                #gpu_index = faiss.GpuIndexFlatL2(res, global_descs[0].size(0))
                cpu_index = faiss.IndexFlatL2(global_descs[0].size(0))

                true_pos_count = 0
                false_pos_count = 0
                distances = []
                gt_positives = []

                for j in range(50, len(global_descs)):
                    cpu_index.add(global_descs[j-50].numpy()[np.newaxis, :])
                    d_torch_gpu, i_torch_gpu = cpu_index.search(global_descs[j].numpy()[np.newaxis, :], 1)
                    index = i_torch_gpu[0][0]
                    distance = float(d_torch_gpu[0])
                    true_pos = pdist[j, index] <= self.max_pos_distance
                    positive = distance < self.pos_threshold

                    #if (pdist[i, index] <= self.max_pos_distance):
                        #print(i, index)
                        #print(pdist[i, index], float(d_torch_gpu[0]))

                    distances.append(distance)
                    gt_positives.append(true_pos)

                    if positive:
                        if true_pos:
                            print(j, index)
                            print(pdist[j, index], distance)

                        if true_pos:
                            true_pos_count += 1
                        else:
                            false_pos_count += 1
                plot_hist(distances, gt_positives, os.path.join(self.log_path, self.sequences[i] + '_lc.png'))
                print(true_pos_count, false_pos_count)

def plot_hist(distances, gt_positives, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='whitegrid')
    df = pd.DataFrame.from_dict({'dist': distances, 'pos': gt_positives})
    
    plt.figure(figsize=(15, 15))
    sns.histplot(data=df, x='dist', hue='pos', multiple='stack', binwidth=0.001, binrange=(0, 0.1))
    plt.savefig(save_path)
    plt.clf()
