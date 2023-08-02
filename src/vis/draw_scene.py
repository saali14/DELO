
import numpy as np
import argparse

import os
import open3d as o3d


def process():
  pcds = []
  for i in range(0, _args.fragments, _args.step): 
    pcd = o3d.io.read_point_cloud(os.path.join(_args.dataset_path, 'cloud_bin_%d.ply'%i)).voxel_down_sample(voxel_size=_args.downsample)
    t = np.load(os.path.join(_args.dataset_path, 'cloud_bin_%d.pose.npy'%i))
    pcd.transform(t)
    pcds.append(pcd)

  if _args.outlier:
    inlier = []
    outlier = []
    for pcd in pcds:
      #_, ids = pcd.remove_radius_outlier(nb_points=5, radius=1)
      _, ids = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=5.0)

      inlier_cloud = pcd.select_by_index(ids)
      outlier_cloud = pcd.select_by_index(ids, invert=True)

      outlier_cloud.paint_uniform_color([1, 0, 0])
      inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

      inlier.append(inlier_cloud)
      outlier.append(outlier_cloud)


      #pcd.paint_uniform_color([1,0,0])
      #colors = np.asarray(pcd.colors)
      #if len(ids) > 0:
      #  np.put(colors, ids, [0.8, .8 , .8])
      #else:
      #  print('!')
      #pcd.colors = o3d.utility.Vector3dVector(colors)
    pcds = inlier + outlier
  o3d.visualization.draw_geometries(pcds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='../../datasets/3DMatch/rgbd_fragments/transformed/sun3d-harvard_c8-hv_c8_3/seq-01')
    parser.add_argument('--fragments', type=int, default=20)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--downsample', type=float, default=0.05)
    parser.add_argument('--outlier', help='show outlier', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
  _args = parse_args()
  process()  