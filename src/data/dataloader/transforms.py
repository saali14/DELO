import time
import plyfile
import numpy as np

#https://github.com/NVIDIA/MinkowskiEngine/issues/330
#import MinkowskiEngine as ME

import torch
#import open3d as o3d
from scipy.spatial.transform import Rotation

#from data.dataloader.sparseThreeDMatchDataModule import collate_pair_fn
from util.pointcloud import transform

from pyntcloud import PyntCloud
import pandas as pd

class Resampler:
    def __init__(self, num: int, upsampling=False):
        """Downsample a point cloud containing N points to one containing M
        Require M <= N.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num
        self.upsampling = upsampling

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])
        sample['points_src'] = self._resample(sample['points_src'], self.num, self.upsampling)
        sample['points_target'] = self._resample(sample['points_target'], self.num, self.upsampling)

        if 'points_neg' in sample:
            sample['points_neg'] = self._resample(sample['points_neg'], self.num, self.upsampling)

        return sample

    @staticmethod
    def _resample(points, k, upsampling):
        """Resamples the points such that there is exactly k points.
        """
        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif k == points.shape[0] or not upsampling:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]

class FixedResampler(Resampler):
    """Fixed resampling to always choose points from evenly spaced indices.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k, upsampling=False):
        resampled = points[np.linspace(0, points.shape[0], num=k, endpoint=False, dtype=int), :]
        return resampled

class VoxelCentroidResampler(Resampler):
    def __init__(self, num: int, n_voxel=(80,80,20)):
        super().__init__(num)
        if len(n_voxel) != 3:
            raise Exception('N_Voxel must have 3 parameters!')
        self.n_x = n_voxel[0]
        self.n_y = n_voxel[1]
        self.n_z = n_voxel[2]

    def _resample(self, points, k, upsampling=False):
        cloud = points_to_pyntcloud(points)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=self.n_x, n_y=self.n_y, n_z=self.n_z)
        new_cloud = cloud.get_sample("voxelgrid_centroids", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
        return new_cloud.points.to_numpy()[:, :3]

class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'src_color' in sample:
            src_perm = np.random.permutation(sample['points_src'].shape[0])
            tgt_perm = np.random.permutation(sample['points_target'].shape[0])
            sample['points_src'] = sample['points_src'][src_perm, :]
            sample['points_target'] = sample['points_target'][tgt_perm, :]
            sample['src_color'] = sample['src_color'][src_perm, :]
            sample['tgt_color'] = sample['tgt_color'][tgt_perm, :]
        else:
            sample['points_target'] = np.random.permutation(sample['points_target'])
            sample['points_src'] = np.random.permutation(sample['points_src'])
        return sample

class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample

"""
class SparseQuantize:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, sample):
        xyz0 = sample['points_src']
        xyz1 = sample['points_target']

        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

        if isinstance(xyz0, torch.Tensor):
            sample['points_src'] = xyz0[sel0]
            sample['points_target'] = xyz1[sel1]
        else:
            sample['points_src'] = torch.from_numpy(xyz0[sel0])
            sample['points_target'] = torch.from_numpy(xyz1[sel1])

        return sample
"""

class AddCoordinates:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, sample):
        xyz0 = sample['points_src']
        xyz1 = sample['points_target']

        sample['coords0'] = torch.floor(xyz0 / self.voxel_size).int()
        sample['coords1'] = torch.floor(xyz1 / self.voxel_size).int()

        return sample

class AddFeatures:
    def __init__(self, features='ones'):
        self.features = features

    def __call__(self, sample):
        xyz0 = sample['points_src']
        xyz1 = sample['points_target']

        if self.features == 'ones':
            sample['feats0'] = torch.ones(xyz0.shape[0], 1)
            sample['feats1'] = torch.ones(xyz1.shape[0], 1)
        elif self.features == 'coords':
            sample['feats0'] = xyz0
            sample['feats1'] = xyz1
        else:
            raise NotImplementedError

        return sample

class MatchingIndices:
    def __init__(self, search_voxel_size):
        self.search_voxel_size = search_voxel_size

    def __call__(self, sample):
        xyz0 = sample['points_src']
        xyz1 = sample['points_target']
        #pcd0 = o3d.geometry.PointCloud()
        #pcd0.points = o3d.utility.Vector3dVector(xyz0)
        #pcd1 = o3d.geometry.PointCloud()
        #pcd1.points = o3d.utility.Vector3dVector(xyz1)

        #pcd0.transform(sample['transform_gt'])
        #pcd_tree = o3d.geometry.KDTreeFlann(pcd1)

        match_inds = []
        for i, point in enumerate(xyz0):
            #[_, idx, _] = pcd_tree.search_radius_vector_3d(point, self.search_voxel_size)
            [_, idx, _] = np.random.randint(3) # pcd_tree.search_radius_vector_3d(point, self.search_voxel_size)
            for j in idx:
                match_inds.append((i, j))

        sample['correspondences'] = match_inds
        return sample

"""
class DGRRequiredInputs:
    #Add required dict parameter for DeepGlobalRegistration, if collate function not used
    def __call__(self, sample):
        return collate_pair_fn([sample])
"""

class AddLatticeRepresentation:
    def __init__(self, data_gen):
        self.data_gen = data_gen

    def __call__(self, sample):
        xyz0 = sample['points_src']
        xyz1 = sample['points_target']
        # 3 layer for rigid network
        generated_data = self.data_gen.compute_generated_data(xyz0.squeeze(), xyz1.squeeze(), 3)
        sample['gen_data'] = generated_data
        return sample

class AddBatchDimension():
    def __call__(self, sample):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                sample[k] = np.expand_dims(v, 0)
        return sample

class RandomTransformSE3:
    def __init__(self, xy_rot_mag: float = 180.0, xy_trans_mag: float = 1.0, z_rot_mag: float = 180.0, z_trans_mag: float = 1.0, apply_transform_prob = 1.0):
        """Applies a random rigid transformation to the source point cloud
        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self.xy_rot_mag = xy_rot_mag
        self.xy_trans_mag = xy_trans_mag
        self.z_rot_mag = z_rot_mag
        self.z_trans_mag = z_trans_mag
        self.apply_transform_prob = apply_transform_prob

    def generate_transform(self):
        # Generate rotation
        anglex = np.random.uniform(-self.xy_rot_mag, self.xy_rot_mag)
        angley = np.random.uniform(-self.xy_rot_mag, self.xy_rot_mag)
        anglez = np.random.uniform(-self.z_rot_mag, self.z_rot_mag)

        t_xy = np.random.uniform(-self.xy_trans_mag, self.xy_trans_mag, 2)
        t_z = np.random.uniform(-self.z_trans_mag, self.z_trans_mag)

        T = np.eye(4, dtype=np.float32)

        R = Rotation.from_euler('xyz', [anglex, angley, anglez], degrees=True).as_matrix()

        T[:3, :3] = R
        T[:2, 3] = t_xy
        T[2, 3] = t_z

        return T, np.linalg.inv(T)

    def apply_transform(self, points, trans_mat):
        points_t = ((trans_mat[:3, :3] @ points.T) + trans_mat[:3, 3, np.newaxis]).T
        return points_t

    def transform(self, points):
        transform_mat, inv_transform = self.generate_transform()
        if isinstance(points, torch.Tensor):
            transform_mat = torch.tensor(transform_mat, device=points.device)
            inv_transform = torch.tensor(inv_transform, device=points.device)
            return transform(points, transform_mat), transform_mat, inv_transform

        return self.apply_transform(points, transform_mat), transform_mat, inv_transform

    def __call__(self, sample):
        if self.apply_transform_prob == 1 or np.random.random() < self.apply_transform_prob:
            if 'deterministic' in sample and sample['deterministic']:
                np.random.seed(sample['idx'])

            sample['points_src'], trans_mat, inv_transform = self.transform(sample['points_src'])
            if 'transform_gt' in sample:
                sample['transform_gt'] = sample['transform_gt'] @ inv_transform

            if 'points_neg' in sample:
                sample['points_neg'], _, _ = self.transform(sample['points_neg'])

        return sample

class AddRPSRNetRepresentation:
    def __init__(self, depth=5, voxel_depth=2, masses=False):
        self.depth = depth
        self.voxel_depth = voxel_depth
        self.masses = masses

    def __call__(self, sample):
        src = sample['points_src']
        tgt = sample['points_target']

        #start = time.time()
        src_x, src_y, src_z, src_labels, src_neighbors, src_src, src_labelsvoxel, src_masses = compute_rpsrnet_tree(src, self.depth, self.voxel_depth, self.masses)
        #print(time.time() - start)
        tgt_x, tgt_y, tgt_z, tgt_labels, tgt_neighbors, tgt_src, tgt_labelsvoxel, tgt_masses = compute_rpsrnet_tree(tgt, self.depth, self.voxel_depth, self.masses)


        sample['src_tree'] = {
            'x': src_x,
            'y': src_y,
            'z': src_z,
            'labels': src_labels,
            'neighbors': src_neighbors,
            'src': src_src,
            'labelsvoxel': src_labelsvoxel
        }

        sample['tgt_tree'] = {
            'x': tgt_x,
            'y': tgt_y,
            'z': tgt_z,
            'labels': tgt_labels,
            'neighbors': tgt_neighbors,
            'src': tgt_src,
            'labelsvoxel': tgt_labelsvoxel
        }

        if self.masses:
            sample['src_tree']['masses'] = src_masses
            sample['tgt_tree']['masses'] = tgt_masses

        #print(time.time() - start)

        return sample

class PadRPSRNetRepresentation:
    def __init__(self, depth=5, masses=False):
        self.depth = depth
        self.masses = masses

    def __call__(self, sample):
        label_max_size = [1, 8, 64, 256, 856, 2544, 6216, 20000, 58224]
        nodes_max_size = [1, 8, 32, 107, 318, 777, 2500, 7278, 17841]
    
        sample['src_tree']['x'] = np.pad(sample['src_tree']['x'], (0, nodes_max_size[self.depth] - sample['src_tree']['x'].shape[0]))
        sample['src_tree']['y'] = np.pad(sample['src_tree']['y'], (0, nodes_max_size[self.depth] - sample['src_tree']['y'].shape[0]))
        sample['src_tree']['z'] = np.pad(sample['src_tree']['z'], (0, nodes_max_size[self.depth] - sample['src_tree']['z'].shape[0]))

        sample['tgt_tree']['x'] = np.pad(sample['tgt_tree']['x'], (0, nodes_max_size[self.depth] - sample['tgt_tree']['x'].shape[0]))
        sample['tgt_tree']['y'] = np.pad(sample['tgt_tree']['y'], (0, nodes_max_size[self.depth] - sample['tgt_tree']['y'].shape[0]))
        sample['tgt_tree']['z'] = np.pad(sample['tgt_tree']['z'], (0, nodes_max_size[self.depth] - sample['tgt_tree']['z'].shape[0]))

        for i in range(0, self.depth + 1):
            sample['src_tree']['labels'][i] = np.pad(sample['src_tree']['labels'][i], (0, label_max_size[i] - sample['src_tree']['labels'][i].shape[0]))
            sample['src_tree']['neighbors'][i] = np.pad(sample['src_tree']['neighbors'][i], ((0, nodes_max_size[i] - sample['src_tree']['neighbors'][i].shape[0]), (0, 0)))

            sample['tgt_tree']['labels'][i] = np.pad(sample['tgt_tree']['labels'][i], (0, label_max_size[i] - sample['tgt_tree']['labels'][i].shape[0]))
            sample['tgt_tree']['neighbors'][i] = np.pad(sample['tgt_tree']['neighbors'][i], ((0, nodes_max_size[i] - sample['tgt_tree']['neighbors'][i].shape[0]), (0, 0)))

            if self.masses:
                sample['src_tree']['masses'][i] = np.pad(sample['src_tree']['masses'][i], (0, nodes_max_size[i] - sample['src_tree']['masses'][i].shape[0]))
                sample['tgt_tree']['masses'][i] = np.pad(sample['tgt_tree']['masses'][i], (0, nodes_max_size[i] - sample['tgt_tree']['masses'][i].shape[0]))



        return sample

def points_to_pyntcloud(points):
    return PyntCloud(pd.DataFrame(data=points, columns=['x', 'y', 'z']))

def pyntcloud_to_points(cloud):
    return cloud.points.to_numpy()[:, :3]

def compute_rpsrnet_tree(points, depth, voxel_depth, use_mass):
    nodes_max_size = [1, 8, 64, 294, 1217, 2046, 2048]

    import rpsrnet_data
    tree = rpsrnet_data.compute_tree(points, depth)
    nodes = np.array(tree.pos[depth]).astype('float32').reshape((-1, 3))

    # nodes = np.vstack([nodes, np.zeros([nodes_max_size[depth] - nodes.shape[0], nodes.shape[1]])]).astype('float32') if batch else nodes.astype('float32')

    labels = []
    neighbors = []
    masses = []

    for i in range(0, depth + 1):
        labeltemp = np.array(tree.label[i]).astype('int64')
        # labeltemp = np.hstack([labeltemp, np.zeros(label_max_size[i] - labeltemp.shape[0])]).astype('int64') if batch else labeltemp.astype('int64')
        labels.append(labeltemp)

        neighbortemp = np.array(tree.neighbor[i]).astype('int64').reshape((-1, 27))
        # neighbortemp = np.vstack([neighbortemp, np.zeros([nodes_max_size[i] - neighbortemp.shape[0], neighbortemp.shape[1]])]).astype('int64') if batch else neighbortemp.astype('int64')
        neighbors.append(neighbortemp)

        masstemp = []
        if use_mass:
            masstemp = np.array(tree.mass[i])
            # masstemp = np.hstack([masstemp, np.zeros(nodes_max_size[i] - masstemp.shape[0])]).astype('float32') if batch else masstemp.astype('float32')
        masses.append(masstemp)

    labels2 = np.array(tree.label[2])
    labels1 = np.array(tree.label[1])
    if (labels2.shape[0] != 64):
        labelsvoxel2 = np.zeros(64).astype('int64')
        for i in range(0, labels1.shape[0]):
            index = labels1[i] - 1
            if (index > 0):
                labelsvoxel2[i * 8: i * 8 + 8] = labels2[index * 8:index * 8 + 8]
    else:
        labelsvoxel2 = labels2.astype('int64')

    if (voxel_depth == 3):
        labels3 = np.array(tree.label[3])
        labels2temp = labelsvoxel2
        # print("Hello", labels3.shape)
        if (labels3.shape[0] != 512):
            labelsvoxel3 = np.zeros(512).astype('int64')
            for i in range(0, labels2temp.shape[0]):
                index = labels2temp[i] - 1
                if (index > 0):
                    labelsvoxel3[i * 8: i * 8 + 8] = labels3[index * 8:index * 8 + 8]
        else:
            labelsvoxel3 = labels3.astype('int64')

    labelsvoxel = labelsvoxel2 if voxel_depth == 2 else labelsvoxel3
    # node2 = np.column_stack([data['Pos_2'][:, 0], data['Pos_2'][:, 1], data['Pos_2'][:, 2]]).T
    if (voxel_depth == 2):
        src = np.array(tree.pos[2]).astype('float32').reshape((-1, 3))
        src = np.vstack([src, np.zeros([nodes_max_size[2] - src.shape[0], src.shape[1]])]).astype(
            'float32').T
    elif (voxel_depth == 3):
        src = np.array(tree.pos[3]).astype('float32').reshape((-1, 3))
        src = np.vstack([src, np.zeros([nodes_max_size[3] - src.shape[0], src.shape[1]])]).astype(
            'float32').T
    # print(labelsvoxel.shape, src.shape)
    return nodes[:, 0].astype('float32'), nodes[:, 1].astype('float32'), nodes[:, 2].astype(
        'float32'), labels, neighbors, src, labelsvoxel, masses