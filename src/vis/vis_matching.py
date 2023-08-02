import numpy as np
import open3d as o3d
import os

points = (np.random.rand(1000, 3) - 0.5) / 4
colors = np.random.rand(1000, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)


#o3d.web_visualizer.draw(pcd)
#o3d.visualization.draw([{'name': 'pcd', 'geometry': pcd}], show_ui=True)



n_points = 1024

import torchvision
from model.dslam.dslam import DSLAM
from data.dataloader import transforms as Transforms

import torch

model_ckpt = '../../results/weights/Kitti/dslam/115_best/epoch=68-step=49265.ckpt'

dataset_path = '/media/skali/WD/dataset/KITTI_odom/dataset'


model_ckpt = os.path.abspath(model_ckpt)

model = DSLAM.load_from_checkpoint(checkpoint_path=model_ckpt, train_only_descriptor=False)
transform = torchvision.transforms.Compose([Transforms.SetDeterministic(),
                Transforms.ShufflePoints(),
                Transforms.VoxelCentroidResampler(n_points),
                Transforms.Resampler(n_points, upsampling=True)])

model.to('cuda')
model.eval()

torch.set_grad_enabled(False)



from torch.utils.data import DataLoader

from data.dataloader.kittiDataset import KittiDataset

dataset = KittiDataset(dataset_path, subset='test', transform=transform, sequence='00', add_pose=True, skipped_frames=[0])
dataloader = DataLoader(dataset, batch_size=None, num_workers=4)



from torch.utils.data._utils.collate import default_collate



sample_num = 100  # try 0, 100, 500


sample = dataloader.dataset.__getitem__(sample_num)

t_sample = transform(sample)
src_points = t_sample['points_src'].squeeze()
tgt_points = t_sample['points_target'].squeeze()

batch = default_collate([t_sample])
pred, info = model.predict_step(batch, 0)
src_embedding = info['src_embedding_t'].detach().cpu().numpy().squeeze()
color = src_embedding.max(axis=0)
tgt_embedding = info['tgt_embedding_t'].detach().cpu().numpy().squeeze()
color = tgt_embedding.max(axis=0)

import matplotlib.cm as cm
from matplotlib.colors import Normalize

def transform_points(points, trans_mat):
    return (trans_mat[:3, :3] @ points.T + trans_mat[:3, 3][:, np.newaxis]).T

cmap = cm.get_cmap('viridis')
norm = Normalize(vmin=0, vmax=max(src_embedding.max(), tgt_embedding.max()))

src_colors = cmap(norm(src_embedding.max(axis=0)))
tgt_colors = cmap(norm(tgt_embedding.max(axis=0)))

#print(src_colors)


shift = False
if shift:
    src_points[:, 1] += 50
    tgt_points[:, 1] -= 50


gt = t_sample['transform_gt'].squeeze()

src_pcd_misaligned = o3d.geometry.PointCloud()
src_pcd_misaligned.points = o3d.utility.Vector3dVector(src_points)
src_pcd_misaligned.colors = o3d.utility.Vector3dVector(src_colors[:, :3])

apply_transform = True

if apply_transform:
    src_points = transform_points(src_points, gt)


src_pcd = o3d.geometry.PointCloud()
src_pcd.points = o3d.utility.Vector3dVector(src_points)
src_pcd.colors = o3d.utility.Vector3dVector(src_colors[:, :3])

tgt_pcd = o3d.geometry.PointCloud()
tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)
tgt_pcd.colors = o3d.utility.Vector3dVector(tgt_colors[:, :3])


import math
scores = torch.softmax((torch.matmul(info['src_embedding_t'][0].T, info['tgt_embedding_t'][0]) / math.sqrt(info['src_embedding_t'][0].size(0))), dim=1).detach().cpu().numpy()
scores = (scores / np.max(scores)) #* n_points
#print(scores)
print(scores.max())
#print(np.concatenate((src_points, tgt_points)).shape)
#line_set.lines = o3d.utility.Vector2iVector(lines)
#print(np.transpose((scores > 0.5).nonzero()))
#print((scores > 0.1).nonzero())


thresholds = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
cur_threshold = 0
threshold = thresholds[cur_threshold]
dcp_line_set = o3d.geometry.LineSet()
dcp_line_set.points = o3d.utility.Vector3dVector(np.concatenate((src_points, tgt_points)))

lines = []
colors = []

for s,t in zip(*((scores > threshold).nonzero())):
    lines.append([s, t + n_points])
    colors.append([1, 0, 0])
dcp_line_set.lines = o3d.utility.Vector2iVector(lines)
dcp_line_set.colors = o3d.utility.Vector3dVector(colors)

print(len(lines))
transport = info['transport'].detach().cpu().numpy().squeeze()
print(transport.sum())
transport = transport / np.max(transport)
print(transport.max())
#print(transport.max(axis = 0))
pot_line_set = o3d.geometry.LineSet()
pot_line_set.points = o3d.utility.Vector3dVector(np.concatenate((src_points, tgt_points)))

lines = []
colors = []

for s,t in zip(*((transport > threshold).nonzero())):
    lines.append([s, t + n_points])
    colors.append([0, 1, 0])
pot_line_set.lines = o3d.utility.Vector2iVector(lines)
pot_line_set.colors = o3d.utility.Vector3dVector(colors)
print(len(lines))


def change_threshold(vis):
    """
    Iterate over thresholds
    """
    global src_points, tgt_points, scores, transport, cur_threshold, thresholds

    dcp_vis = vis.scene.has_geometry('dcp_matches') and vis.get_geometry('dcp_matches').is_visible
    pot_vis = vis.scene.has_geometry('pot_matches') and vis.get_geometry('pot_matches').is_visible

    vis.remove_geometry('dcp_matches')
    vis.remove_geometry('pot_matches')

    cur_threshold = (cur_threshold + 1) % len(thresholds)
    threshold = thresholds[cur_threshold]


    dcp_line_set = o3d.geometry.LineSet()
    dcp_line_set.points = o3d.utility.Vector3dVector(np.concatenate((src_points, tgt_points)))

    lines = []
    colors = []

    for s, t in zip(*((scores > threshold).nonzero())):
        lines.append([s, t + n_points])
        colors.append([1, 0, 0])
    dcp_line_set.lines = o3d.utility.Vector2iVector(lines)
    dcp_line_set.colors = o3d.utility.Vector3dVector(colors)

    pot_line_set = o3d.geometry.LineSet()
    pot_line_set.points = o3d.utility.Vector3dVector(np.concatenate((src_points, tgt_points)))

    lines = []
    colors = []

    for s, t in zip(*((transport > threshold).nonzero())):
        lines.append([s, t + n_points])
        colors.append([0, 1, 0])
    pot_line_set.lines = o3d.utility.Vector2iVector(lines)
    pot_line_set.colors = o3d.utility.Vector3dVector(colors)

    print('Threshold %.04f' % threshold)

    vis.add_geometry({'name': 'dcp_matches', 'geometry': dcp_line_set, 'is_visible': dcp_vis})
    vis.add_geometry({'name': 'pot_matches', 'geometry': pot_line_set, 'is_visible': pot_vis})

o3d.visualization.draw([{'name': 'src', 'geometry': src_pcd}, {'name': 'tgt', 'geometry': tgt_pcd}, {'name': 'dcp_matches', 'geometry': dcp_line_set}, {'name': 'pot_matches', 'geometry': pot_line_set}], show_ui=True, show_skybox=False, bg_color=(0, 0, 0, 0), actions=[['next threshold', change_threshold]])

