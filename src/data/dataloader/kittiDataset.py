import numpy as np
import logging
import os

import pickle as pkl
from util.kittiutil import velo_to_cam_trans
from pathlib import Path
from random import Random
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'train', transform=None, skipped_frames=(0,1,2,3,4), seed=42, lc_sample=False, lc_neg_min_dist=4.0, lc_neg_min_skipped=50, lc_rand_src=False, sequence=None, add_pose=False, **kwargs):
        """Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self._logger = logging.getLogger('core.data')
        self._root = dataset_path
        self._logger.info('Loading data for {}'.format(subset))

        if not os.path.exists(os.path.join(dataset_path)):
            raise Exception("Dataset not found at: {}".format(os.path.join(dataset_path)))
        if sequence is None:
            self._sequences = self.load_sequences(subset)
        else:
            self._sequences = [sequence]

        self.random = Random(seed)
        self.samples = []
        self.subset = subset
        self.skipped_frames = skipped_frames
        self.add_pose = add_pose
        self.lc_sample = lc_sample
        self.lc_neg_min_dist = lc_neg_min_dist
        self.lc_neg_min_skipped = lc_neg_min_skipped
        self.lc_rand_src = lc_rand_src

        if self.lc_sample:
            self.distances = {}

        for sequence in self._sequences:
            pose_file = os.path.join(self._root, 'sequences', sequence, 'poses.txt')
            poses = self.load_poses(pose_file)
            velo_to_cam_t = velo_to_cam_trans[sequence]
            n_frames = len(poses)

            if self.subset == 'train':
                targets = np.load(os.path.join(self._root, 'sequences', sequence, 'train.npy'), allow_pickle=False)
            elif self.subset == 'val':
                targets = np.load(os.path.join(self._root, 'sequences', sequence, 'test.npy'), allow_pickle=False)
            else:
                targets = np.arange(n_frames - 1)

            for i in targets:
                if self.lc_rand_src:
                    js = None
                    gt = None
                    if i >= n_frames - 1:   # skip last frame
                        continue
                else:
                    js = [i + 1 + j for j in self.skipped_frames if (i + 1 + j) < n_frames]
                    gt = [(np.linalg.inv(poses[i] @ velo_to_cam_t) @ poses[j] @ velo_to_cam_t).astype('float32') for j in js]

                    if len(js) == 0:
                        continue

                sample = {
                    'sequence': sequence,
                    'srcs': js,
                    'target': i,
                    'gt': gt
                }

                if self.add_pose:
                    sample['pose'] = poses[i] @ velo_to_cam_t
                self.samples.append(sample)

            if self.lc_sample:
                poses_array = np.array(poses)
                Ts = poses_array[:, :3, 3]
                pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
                pdist = np.sqrt(pdist.sum(-1))

                # use only target frames from subset as negative samples, as setting distance to zero for all other samples
                mask = np.ones(pdist.shape[0], dtype=bool)
                mask[targets] = False
                pdist[:, mask] = -1

                self.distances[sequence] = pdist

        self._logger.info('Loaded {} {} instances.'.format(len(self.samples), subset))
        self._transform = transform

    def __getitem__(self, item):
        t = self.samples[item]

        # lazy load
        if self.lc_rand_src and t['srcs'] is None:
            t['srcs'] = np.where((self.distances[t['sequence']][t['target']] <= self.lc_neg_min_dist) & (self.distances[t['sequence']][t['target']] > 0))[0]
            if len(t['srcs']) == 0:
                t['srcs'] = [t['target'] + 1]

        src_index = self.random.randrange(len(t['srcs']))
        src_id = t['srcs'][src_index]

        src_file = os.path.join(self._root, 'sequences', t['sequence'], 'velodyne', '%06d.bin' % src_id)
        tgt_file = os.path.join(self._root, 'sequences', t['sequence'], 'velodyne', '%06d.bin' % t['target'])

        src = np.fromfile(src_file, dtype=np.float32).reshape((-1, 4))[:, :3]
        tgt = np.fromfile(tgt_file, dtype=np.float32).reshape((-1, 4))[:, :3]

        if self.lc_rand_src:
            sample = {'points_target': tgt, 'points_src': src,
                      'label': t['sequence'],
                      'idx': np.array(item, dtype=np.int32),
                      'src_id': src_id, 'target_id': t['target']}
        else:
            sample = {'points_target': tgt, 'points_src': src,
                      'transform_gt': t['gt'][src_index], 'label': t['sequence'],
                      'idx': np.array(item, dtype=np.int32),
                      'src_id': src_id, 'target_id': t['target']}

        if self.add_pose:
            sample['pose'] = t['pose']
        if self.lc_sample:
            anchor_id = t['target']
            lc_candidates = np.where(self.distances[t['sequence']][anchor_id] > self.lc_neg_min_dist)[0]
            neg_id = self.random.choice([i for i in lc_candidates if abs(i - anchor_id) > self.lc_neg_min_skipped])
            sample['neg_id'] = neg_id
            neg_file = os.path.join(self._root, 'sequences', t['sequence'], 'velodyne', '%06d.bin' % neg_id)
            neg_points = np.fromfile(neg_file, dtype=np.float32).reshape((-1, 4))[:, :3]
            sample['points_neg'] = neg_points
        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        #return 100
        return len(self.samples)

    @property
    def sequences(self):
        return self._sequences

    @staticmethod
    def load_sequences(subset):
        fname = Path(__file__).parent.parent/'config'/(subset + '_kitti.txt')
        with open(fname) as f:
            lines = f.readlines()
        return [x.strip() for x in lines]

    @staticmethod
    def load_poses(file):
        poses = []
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            pose = np.fromstring(line, dtype=float, sep=' ')
            pose = pose.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
        return poses
