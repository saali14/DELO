import copy
import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torchvision
from open3d.visualization import draw

from data.dataloader import ThreeDMatchDataModule, KittiDataModule
from data.dataloader.transforms import RandomTransformSE3
from model import add_model_specific_args, load_model
from util.file_helper import create_dir

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

models = {
    'ICP': {
        'name': 'ICP',
        'model': 'icp',
        'color': [0, 1, 1],
    },
    'DCP': {
        'name': 'DCP',
        'model': 'dcp',
        'checkpoint': '/media/skali/My Passport/new_logs/logs/train/dcp/version_31/checkpoint/epoch=73-step=105745.ckpt',
        'color': [0, 1, 0],
    },
    'DSLAM': {
        'name': 'DSLAM',
        'model': 'dslam',
        'checkpoint': '/media/skali/My Passport/eval_logs/dslam/version_115/checkpoint/epoch=68-step=49265.ckpt',
        'color': [0.5, 0, 0.5],
    },
}

# currently rendered geometry dicts
drawn_pcds = []

# point colors
SRC_COLOR = [1, 0, 0]
TGT_COLOR = [0, 0, 1]
GT_COLOR = [1, 0, 1]

batch = None
results = {}
dl_iter = None

uniform_color = True
args = None
random_trans = False
transform = RandomTransformSE3(xy_rot_mag=3.0, xy_trans_mag=1.5, z_rot_mag=180.0, z_trans_mag=0.25, apply_transform_prob=0.8)

def main():
    global batch, dl_iter, drawn_pcds, args, uniform_color, random_trans
    parser = ArgumentParser()
    parser.add_argument('--train_data_dir',
                        type=str, metavar='PATH',
                        default='/media/skali/My Passport/Kitti/data_odometry_velodyne/dataset',
                        help='path to the processed dataset. Default: ../datasets/3DMatch/3DMatch_5cm/training/')
    parser.add_argument('--test_data_dir',
                        type=str, metavar='PATH',
                        default='/media/skali/My Passport/Kitti/data_odometry_velodyne/dataset',
                        help='path to the processed dataset. Default: ../datasets/3DMatch/Geometric_Registration_Benchmark/')
    parser.add_argument('--dataset', default='train', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--weights_path', default='../weights', type=str, metavar='PATH',
                        help='path to folder with model checkpoints')
    parser.add_argument('--out_path', default='../out/kitti2', type=str, metavar='PATH',
                        help='path to save')
    parser.add_argument('--random_trans', action='store_true')

    # add model specific arguments to parser
    #for _, model in models.items():
    #    parser = add_model_specific_args(model['model'], parser)

    args = parser.parse_args()

    random_trans = args.random_trans

    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None}  # remove None entries

    # visualizer actions -> buttons in interface
    actions = [
        ['next batch', next_batch],
        ['save', save],
        ['gt', gt_transform],
        ['color', set_color],
    ]

    # load network models
    for _, model in models.items():
        try:
            print('Load model {}'.format(model['name']))
            if 'checkpoint' in model:
                path = os.path.join(model['checkpoint'])
                #path = os.path.join(args.weights_path, model['checkpoint'])
                _model = load_model(model['model'], path, dict_args)
            else:
                _model = load_model(model['model'], False, dict_args)
            _model.to('cuda')
            _model.eval()
            actions.append(create_forward_action(model['name'], _model, model['color']))
        except FileNotFoundError:
            print('Could not load checkpoint for model {} from file {}'.format(model['name'], path))

    if args.dataset == 'val':
        dm = KittiDataModule(val_batch_size=None, train_batch_size=None, color=True, **dict_args)
        dm.setup(stage='fit')
        dl_iter = dm.val_dataloader().__iter__()
    elif args.dataset == 'test':
        dm = KittiDataModule(val_batch_size=None, train_batch_size=None, color=False, **dict_args)
        dm.setup(stage='test')
        dl_iter = dm.test_dataloader()[5].__iter__()
    # default train case
    else:
        dm = KittiDataModule(val_batch_size=None, train_batch_size=None, color=True, **dict_args)
        dm.setup(stage='fit')
        dl_iter = dm.train_dataloader().__iter__()

    # show first sample
    batch = next(dl_iter)
    print_curr_batch_infos()
    geometries = process_batch()
    drawn_pcds += geometries
    draw(geometries, show_ui=True, actions=actions)


def next_batch(vis):
    """
    Load and show next sample
    """
    global batch, dl_iter, drawn_pcds, uniform_color, random_trans

    batch = next(dl_iter)
    if random_trans:
        batch = transform(batch)

    geometries = process_batch()
    print_curr_batch_infos()



    clear_geometry(vis)
    drawn_pcds.clear()
    for g in geometries:
        vis.add_geometry(g)
        drawn_pcds.append(g)


def save(vis):
    global batch, args, result

    src_pcd = to_o3d(batch['points_src'].numpy().squeeze())
    save_pcd(src_pcd, 'src_' + str(batch['src_id'].cpu().numpy()) + '.ply')
    save_pcd(to_o3d(batch['points_target'].numpy().squeeze()), 'tgt_' + str(batch['target_id'].cpu().numpy()) + '.ply')

    src_pcd_t = transform_pcd(src_pcd, batch['transform_gt'].squeeze())
    save_pcd(src_pcd_t, 'src_t_' + str(batch['src_id'].cpu().numpy()) + '.ply')

    for k,v in results.items():
        if v['src_id'] == int(batch['src_id'].cpu().numpy()) and v['target_id'] == int(batch['target_id'].cpu().numpy()):

            src_color, tgt_color = None, None
            if 'src_feat' in v:
                tsne = embed_tsne(np.concatenate([v['src_feat'], v['tgt_feat']]))
                src_color = get_color_map(tsne[:(tsne.shape[0] // 2)])
                tgt_color = get_color_map(tsne[(tsne.shape[0] // 2):])



            src_pcd_down = to_o3d(v['points_src'], color=src_color)
            save_pcd(src_pcd_down, 'src_down_' + k + '_' + str(v['src_id']) + '.ply')
            save_pcd(to_o3d(v['points_target'], color=tgt_color), 'tgt_down_' + k + '_' + str(v['target_id']) + '.ply')

            src_pcd_down_t = transform_pcd(src_pcd_down, batch['transform_gt'].squeeze())
            save_pcd(src_pcd_down_t, 'src_down_t_' + k + '_' + str(v['src_id']) + '.ply')

            src_pcd_pred = transform_pcd(src_pcd, v['pred'])
            save_pcd(src_pcd_pred, 'src_t_pred_' + k + '_' + str(v['src_id']) + '.ply')

            src_pcd_down_pred = transform_pcd(src_pcd_down, v['pred'])
            save_pcd(src_pcd_down_pred, 'src_down_pred_' + k + '_' + str(v['src_id']) + '.ply')





def save_pcd(pcd, name):
    global args, batch
    create_dir(os.path.join(args.out_path, batch['label'], str(batch['target_id'].cpu().numpy())))
    o3d.io.write_point_cloud(os.path.join(args.out_path, batch['label'], str(batch['target_id'].cpu().numpy()), name), pcd, write_ascii=True)

def to_o3d(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def transform_pcd(pcd, T):
    pcd_t = copy.deepcopy(pcd)
    return pcd_t.transform(T)



def get_color_map(x):
  colours = plt.cm.Spectral(x)
  return colours[:, :3]

def embed_tsne(data):
  """
  N x D np.array data
  """
  tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
  tsne_results = tsne.fit_transform(data)
  tsne_results = np.squeeze(tsne_results)
  tsne_min = np.min(tsne_results)
  tsne_max = np.max(tsne_results)
  return (tsne_results - tsne_min) / (tsne_max - tsne_min)


def print_curr_batch_infos():
    global batch
    print('Loaded sample: (sequence: {}, src: {}, target {})'
          .format(batch['label'], batch['src_id'], batch['target_id']))


def clear_geometry(vis):
    """
    Remove all pointclouds from visualizer
    """
    # vis.scene.clear_geometry()   only view, clears not menu
    vis.remove_geometry('src')
    vis.remove_geometry('target')

    # clear possible transformed pcds
    vis.remove_geometry('src_trans')
    vis.remove_geometry('src_gt')

    for k, v in models.items():
        vis.remove_geometry('src_' + v['name'])


def process_batch():
    """
    Add src and target pointcloud from current batch to visualizer
    """
    global batch
    pcd0 = o3d.geometry.PointCloud()
    print(batch['points_src'].numpy().squeeze().shape)
    pcd0.points = o3d.utility.Vector3dVector(batch['points_src'].numpy().squeeze())
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(batch['points_target'].numpy().squeeze())
    if 'src_color' in batch:
        pcd0.colors = o3d.utility.Vector3dVector(batch['src_color'].numpy().squeeze())
        pcd1.colors = o3d.utility.Vector3dVector(batch['tgt_color'].numpy().squeeze())
    else:
        pcd0.paint_uniform_color(SRC_COLOR)
        pcd1.paint_uniform_color(TGT_COLOR)

    return [{
        'name': 'src',
        'geometry': pcd0,
        'color': SRC_COLOR,
        'is_visible': True,
        'color_copy': np.asarray(pcd0.colors).copy(),
    },
        {
            'name': 'target',
            'geometry': pcd1,
            'color': TGT_COLOR,
            'is_visible': True,
            'color_copy': np.asarray(pcd1.colors).copy(),
        }]


def add_transformed(vis, T, name='trans', color=(0, 0, 0)):
    """
    Add transformed src pointcloud from current batch to visualizer
    """
    global batch, drawn_pcds, uniform_color
    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(batch['points_src'].numpy().squeeze())
    if 'src_color' in batch:
        pcd0.colors = o3d.utility.Vector3dVector(batch['src_color'].numpy().squeeze())
    pcd0 = pcd0.transform(T)
    g = {
        'name': 'src_' + name,
        'geometry': pcd0,
        'color': color,
        'is_visible': True,
        'color_copy': np.asarray(pcd0.colors).copy(),
    }
    if uniform_color:
        g['geometry'].paint_uniform_color(g['color'])
    vis.add_geometry(g)
    drawn_pcds.append(g)
    vis.show_geometry('src', False)


def gt_transform(vis):
    """
    Add with ground-truth transformed src pointcloud from current batch to visualizer
    """
    if vis.scene.has_geometry('src_gt'):
        return
    gt = batch['transform_gt'].squeeze()
    add_transformed(vis, gt, 'gt', color=GT_COLOR)


def create_forward_action(name, model, color):
    """
    Add action to visualizer, that runs inference on model and show transformed pointcloud
    """
    transform = torchvision.transforms.Compose(model.get_default_batch_transform(voxel=True))

    def action(vis):
        if vis.scene.has_geometry('src_' + name):
            return
        _batch = copy.deepcopy(batch)
        batch_t = transform(_batch)
        pred, info = model.predict_step(batch_t, 0)
        T = np.identity(4)
        T[:3, :4] = pred.detach().cpu().numpy().squeeze()

        add_transformed(vis, T, name, color=color)

        results[name] = {
            'method': name,
            'src_id': int(batch_t['src_id'].squeeze()),
            'target_id': int(batch_t['target_id'].squeeze()),
            'pred': T,
            'points_src': batch_t['points_src'].squeeze(),
            'points_target': batch_t['points_target'].squeeze(),
        }

        if name == 'DSLAM':
            results[name]['src_feat'] = info['src_embedding_t'].detach().cpu().numpy().squeeze().T
            results[name]['tgt_feat'] = info['tgt_embedding_t'].detach().cpu().numpy().squeeze().T

    return [name + ' transform', action]


def set_color(vis):
    """
    Toggles pointcloud colors. Uniform or original color.
    """
    global batch, drawn_pcds, uniform_color

    uniform_color = not uniform_color  # toggle current option
    update_visibility(vis)
    clear_geometry(vis)
    for g in drawn_pcds:
        if uniform_color:
            g['geometry'].paint_uniform_color(g['color'])
        else:
            g['geometry'].colors = o3d.utility.Vector3dVector(g['color_copy'])

        vis.add_geometry(g)


def update_visibility(vis):
    """
    Update visibility of geometries in geometry dict 'drawn_pcds'
    """
    global drawn_pcds
    for g in drawn_pcds:
        g['is_visible'] = vis.scene.has_geometry(g['name']) and vis.get_geometry(g['name']).is_visible


if __name__ == "__main__":
    main()
