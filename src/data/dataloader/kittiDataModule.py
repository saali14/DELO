import numpy as np
import torch
import torchvision
import logging

from argparse import ArgumentParser
from typing import Optional

from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader

from data.dataloader.kittiDataset import KittiDataset
from data.dataloader import transforms as Transforms


class KittiDataModule(LightningDataModule):
    def __init__(self,
                 train_data_dir='../datasets/Kitti',
                 test_data_dir='../datasets/Kitti',
                 train_batch_size=8,
                 val_batch_size=8,
                 test_batch_size=8,
                 noise_type='clean',
                 num_points=1024,
                 shuffle_train=True,
                 voxel_size=0.2,
                 skipped_frames=(0,1,2,3,4),
                 n_voxel=(80,80,20),
                 depth=5,
                 voxel_depth=2,
                 use_mass=False,
                 **kwargs):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.noise_type = noise_type
        self.num_points = num_points
        self.shuffle_train = shuffle_train
        self.voxel_size = voxel_size
        self.skipped_train_frames = skipped_frames
        self.kwargs = kwargs
        self.n_voxel = n_voxel
        self.depth = depth
        self.voxel_depth = voxel_depth
        self.use_mass = use_mass
        self._logger = logging.getLogger('core.data')

    @staticmethod
    def add_data_specific_args(parent_parser):
        noise_choices=['clean', 'downsample', 'sparse', 'downsample_transform', 'downsample_transform_small', 'voxel_downsample_transform_small', 'voxel_downsample_transform', 'voxel_downsample', 'dgr', 'rpsrnet']
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--train_data_dir', type=str, metavar='PATH', help='path to the processed dataset. Default: ../datasets/Kitti/')
        parser.add_argument('--test_data_dir', type=str, metavar='PATH', help='path to the processed dataset. Default: ../datasets/Kitti/')
        parser.add_argument('--num_points', type=int, metavar='N', help='points in point-cloud (default: 1024)')
        parser.add_argument('--noise_type', choices=noise_choices, help='Types of perturbation to consider')
        parser.add_argument('--train_batch_size', type=int, metavar='N', help='train batch size (default: 8)')
        parser.add_argument('--val_batch_size', type=int, metavar='N', help='batch size for validation and test (default: 16)')
        parser.add_argument('--lc_neg_min_skipped', type=int, help='min number of frames beTestMetrics_ClbLoggertween negative loop closure frame and target (default: 50)')
        parser.add_argument('--lc_neg_min_dist', type=float, help='min distance of negative loop closure frame to target (default: 4.0)')
        parser.add_argument('--lc_sample', action='store_true', help='load negative loop closure frame (default: False)')
        parser.add_argument('--lc_rand_src', action='store_true', help='load src random from true loop candidates (default: False)')
        parser.add_argument('--skipped_frames', type=int, nargs='+')
        parser.add_argument('--n_voxel', type=int, nargs='+')
        parser.add_argument('--depth', type=int)
        parser.add_argument('--voxel_depth', type=int)
        parser.add_argument('--use_mass', action='store_true')
        return parser

    def get_transforms(self):
        val_transforms = None
        if   self.noise_type is None or self.noise_type == "clean":
            train_transforms = [Transforms.ShufflePoints()]
            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.ShufflePoints()]
        elif self.noise_type == "downsample":
            train_transforms = [Transforms.Resampler(self.num_points, upsampling=True),
                                Transforms.ShufflePoints()]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.FixedResampler(self.num_points),
                               Transforms.ShufflePoints()]
        elif self.noise_type == 'sparse':
            train_transforms = [Transforms.ShufflePoints(),
                                Transforms.Resampler(self.num_points),
                                ToSparseTensor(voxel_size=self.voxel_size)]
            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.ShufflePoints(),
                               Transforms.Resampler(self.num_points),
                               ToSparseTensor(voxel_size=self.voxel_size)]
        elif self.noise_type == 'downsample_transform':
            train_transforms = [Transforms.Resampler(self.num_points, upsampling=True),
                                Transforms.ShufflePoints(),
                                Transforms.RandomTransformSE3(xy_rot_mag=3.0, xy_trans_mag=1.5, z_rot_mag=180.0, z_trans_mag=0.25, apply_transform_prob=0.8)]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.FixedResampler(self.num_points),
                               Transforms.ShufflePoints()]
        elif self.noise_type == 'downsample_transform_small':
            train_transforms = [Transforms.Resampler(self.num_points, upsampling=True),
                                Transforms.ShufflePoints(),
                                Transforms.RandomTransformSE3(xy_rot_mag=0.5, xy_trans_mag=1.5, z_rot_mag=2.0, z_trans_mag=0.25, apply_transform_prob=0.8)]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.FixedResampler(self.num_points),
                               Transforms.ShufflePoints()]
        elif self.noise_type == 'voxel_downsample_transform_small':
            train_transforms = [Transforms.VoxelCentroidResampler(self.num_points, n_voxel=self.n_voxel),
                                Transforms.Resampler(self.num_points, upsampling=True),
                                Transforms.ShufflePoints(),
                                Transforms.RandomTransformSE3(xy_rot_mag=0.5, xy_trans_mag=1.5, z_rot_mag=2.0, z_trans_mag=0.25, apply_transform_prob=0.8)]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.VoxelCentroidResampler(self.num_points, n_voxel=self.n_voxel),
                               Transforms.FixedResampler(self.num_points),
                               Transforms.ShufflePoints()]
        elif self.noise_type == 'voxel_downsample_transform':
            train_transforms = [Transforms.VoxelCentroidResampler(self.num_points, n_voxel=self.n_voxel),
                                Transforms.Resampler(self.num_points, upsampling=True),
                                Transforms.ShufflePoints(),
                                Transforms.RandomTransformSE3(xy_rot_mag=3.0, xy_trans_mag=1.5, z_rot_mag=180.0, z_trans_mag=0.25, apply_transform_prob=0.8)]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.VoxelCentroidResampler(self.num_points, n_voxel=self.n_voxel),
                               Transforms.FixedResampler(self.num_points),
                               Transforms.ShufflePoints()]
        elif self.noise_type == "voxel_downsample":
            train_transforms = [Transforms.VoxelCentroidResampler(self.num_points, n_voxel=self.n_voxel),
                                Transforms.Resampler(self.num_points, upsampling=True),
                                Transforms.ShufflePoints()]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.VoxelCentroidResampler(self.num_points, n_voxel=self.n_voxel),
                               Transforms.FixedResampler(self.num_points),
                               Transforms.ShufflePoints()]
        elif self.noise_type == "dgr":
            train_transforms = [Transforms.ShufflePoints(),
                                #Transforms.Resampler(self.num_points),
                                Transforms.SparseQuantize(voxel_size=self.voxel_size),
                                Transforms.AddCoordinates(voxel_size=self.voxel_size),
                                Transforms.AddFeatures(),
                                Transforms.MatchingIndices(search_voxel_size=4*self.voxel_size)]

            val_transforms = [Transforms.SetDeterministic(),
                               Transforms.ShufflePoints(),
                               #Transforms.Resampler(self.num_points),
                               Transforms.SparseQuantize(voxel_size=self.voxel_size),
                               Transforms.AddCoordinates(voxel_size=self.voxel_size),
                               Transforms.AddFeatures(),
                              Transforms.MatchingIndices(search_voxel_size=4*self.voxel_size)]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.ShufflePoints(),
                               Transforms.SparseQuantize(voxel_size=self.voxel_size),
                               Transforms.AddCoordinates(voxel_size=self.voxel_size),
                               Transforms.AddFeatures()]
        elif self.noise_type == "rpsrnet":
            train_transforms = [Transforms.ShufflePoints(),
                                Transforms.AddRPSRNetRepresentation(depth=self.depth, voxel_depth=self.voxel_depth, masses=self.use_mass),
                                Transforms.PadRPSRNetRepresentation(depth=self.depth, masses=self.use_mass),
                                Transforms.Resampler(512, upsampling=True)]

            test_transforms = [Transforms.SetDeterministic(),
                               Transforms.ShufflePoints(),
                               Transforms.AddRPSRNetRepresentation(depth=self.depth, voxel_depth=self.voxel_depth, masses=self.use_mass),
                               Transforms.PadRPSRNetRepresentation(depth=self.depth, masses=self.use_mass),
                               Transforms.Resampler(512, upsampling=True)]
        else:
            raise NotImplementedError

        train_transforms = torchvision.transforms.Compose(train_transforms)
        test_transforms = torchvision.transforms.Compose(test_transforms)
        val_transforms = torchvision.transforms.Compose(val_transforms) if val_transforms is not None else test_transforms
        return train_transforms, test_transforms, val_transforms

    def prepare_data(self):
        # do only on one process, e.g. download dataset
        pass

    def setup(self, stage: Optional[str] = None):
        # do for every gpu
        self.train_transform, self.test_transform, self.val_transforms = self.get_transforms()

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = KittiDataset(self.train_data_dir, subset='train', transform=self.train_transform, skipped_frames=self.skipped_train_frames, **self.kwargs)
            self.val_set = KittiDataset(self.train_data_dir, subset='val', transform=self.val_transforms, skipped_frames=[0], **self.kwargs)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.sequences = KittiDataset.load_sequences('test')
            for sequence in self.sequences:
                self.test_sets.append(KittiDataset(self.test_data_dir, subset='test', transform=self.test_transform, sequence=sequence, add_pose=True, skipped_frames=[0], **self.kwargs))  #TODO skipped frames?

        # Apply no transformations when evaluating dataset
        if stage == 'data-eval' or stage is None:
            self.train_set = KittiDataset(self.train_data_dir, subset='train', transform=None, **self.kwargs)
            self.val_set = KittiDataset(self.train_data_dir, subset='val', transform=None, **self.kwargs)
            self.test_set = KittiDataset(self.test_data_dir, subset='test', transform=None, **self.kwargs)

        self.collate_fn = None
        if self.noise_type == 'sparse':
            from torchsparse.utils.collate import sparse_collate_fn
            self.collate_fn = sparse_collate_fn
        elif self.noise_type == 'dgr':
            from data.dataloader.sparseThreeDMatchDataModule import collate_pair_fn
            self.collate_fn = collate_pair_fn
        elif self.noise_type == 'rpsrnet':
            self.collate_fn = collate_pair_rpsrnet

    def train_dataloader(self):
        self._logger.info('Loading training Set .{}'.format(len(self.train_set)))
        return DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=8, pin_memory=True, shuffle=self.shuffle_train, collate_fn=self.collate_fn, 
            #drop_last=(True if self.train_batch_size is not None else False)
            )

    def val_dataloader(self):
        self._logger.info('Loading validation Set .{}'.format(len(self.val_set)))
        return DataLoader(self.val_set, batch_size=self.val_batch_size, num_workers=8, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        self._logger.info('Loading test Set .{}'.format(len(self.test_set)))
        dataloaders = []
        for test_set in self.test_sets:
            dataloaders.append(DataLoader(test_set, batch_size=self.val_batch_size, num_workers=8, collate_fn=self.collate_fn))

        return dataloaders

class ToSparseTensor:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, sample):
        xyz0 = sample['points_src']
        xyz1 = sample['points_target']

        if isinstance(xyz0, torch.Tensor):
            sample['points_src'] = self._toSparseTensor(xyz0.numpy(), self.voxel_size)
            sample['points_target'] = self._toSparseTensor(xyz1.numpy(), self.voxel_size)
        else:
            sample['points_src'] = self._toSparseTensor(xyz0, self.voxel_size)
            sample['points_target'] = self._toSparseTensor(xyz1, self.voxel_size)
        sample['points_src_copy'] = [xyz0]
        return sample

    @staticmethod
    def _toSparseTensor(pc, voxel_size):
        from torchsparse import SparseTensor
        #from torchsparse.utils.quantize import sparse_quantize
        #rounded_pc = np.round(pc[:, :3] / voxel_size).astype(np.int32)
        coords, inds = sparse_quantize(pc, voxel_size=voxel_size, return_index=True)
        #voxel_pc = rounded_pc[inds]
        voxel_feat = pc[inds]

        #print(np.floor(pc / (0.2, 0.2, 0.2)).astype(np.int32))


        return SparseTensor(coords=coords, feats=voxel_feat)

from itertools import repeat
def sparse_quantize(coords, voxel_size = 1, *, return_index: bool = False, return_inverse: bool = False):
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, 3))
    assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)



    _, indices, inverse_indices = np.unique(ravel_hash(coords),
                                            return_index=True,
                                            return_inverse=True)
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs

def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape
    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h

def collate_pair_rpsrnet(inputs):
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = collate_pair_rpsrnet(
                    [input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = [torch.tensor(input[name]) for input in inputs]
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = [input[name] for input in inputs]
            else:
                output[name] = [input[name] for input in inputs]
            output[name] = stack_rpsrnet_input(output[name], name)
        return output
    else:
        return inputs

def stack_rpsrnet_input(input, name):
    if name in ['transform_gt', 'x', 'y', 'z', 'labels', 'neighbors', 'labelsvoxel', 'src', 'masses', 'points_src', 'points_target']:
        if isinstance(input[0], list):
            return [torch.stack([torch.tensor(input[j][i]) for j in range(len(input))], dim=0) for i in range(len(input[0]))]
        return torch.stack(input, dim=0)
    else:
        return input
