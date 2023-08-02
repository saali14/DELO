import copy
import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torchvision
from open3d.visualization import draw

from data.dataloader import ThreeDMatchDataModule
from model import add_model_specific_args, load_model

models = {
    'DGR': {
        'name': 'DGR',
        'model': 'dgr',
        'checkpoint': 'ResUNetBN2C-feat32-3dmatch-v0.05.pth',
        'color': [0, 1, 0],
    },
}

# currently rendered geometry dicts
drawn_pcds = []

# point colors
SRC_COLOR = [1, 0, 0]
TGT_COLOR = [0, 0, 1]
GT_COLOR = [1, 0, 1]

batch = None
dl_iter = None

uniform_color = False
args = None

def main():
    global batch, dl_iter, drawn_pcds, args, uniform_color
    parser = ArgumentParser()
    parser.add_argument('--train_data_dir',
                        type=str, metavar='PATH',
                        help='path to the processed dataset. Default: ../datasets/3DMatch/3DMatch_5cm/training/')
    parser.add_argument('--test_data_dir',
                        type=str, metavar='PATH',
                        help='path to the processed dataset. Default: ../datasets/3DMatch/Geometric_Registration_Benchmark/')
    parser.add_argument('--dataset', default='train', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--weights_path', default='../weights', type=str, metavar='PATH',
                        help='path to folder with model checkpoints')

    # add model specific arguments to parser
    for _, model in models.items():
        parser = add_model_specific_args(model['model'], parser)

    args = parser.parse_args()
    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None}  # remove None entries

    # visualizer actions -> buttons in interface
    actions = [
        ['next batch', next_batch],
        ['gt', gt_transform],
        ['color', set_color],
    ]

    # load network models
    for _, model in models.items():
        path = os.path.join(args.weights_path, model['checkpoint'])
        try:
            print('Load model {}'.format(model['name']))
            _model = load_model(model['model'], path, dict_args)
            _model.to('cuda')
            _model.eval()
            actions.append(create_forward_action(model['name'], _model, model['color']))
        except FileNotFoundError:
            print('Could not load checkpoint for model {} from file {}'.format(model['name'], path))

    if args.dataset == 'val':
        dm = ThreeDMatchDataModule(val_batch_size=None, train_batch_size=None, color=True, **dict_args)
        dm.setup(stage='fit')
        dl_iter = dm.val_dataloader().__iter__()
    elif args.dataset == 'test':
        dm = ThreeDMatchDataModule(val_batch_size=None, train_batch_size=None, color=False, **dict_args)
        dm.setup(stage='test')
        dl_iter = dm.test_dataloader().__iter__()
        uniform_color = True
    # default train case
    else:
        dm = ThreeDMatchDataModule(val_batch_size=None, train_batch_size=None, color=True, **dict_args)
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
    global batch, dl_iter, drawn_pcds, uniform_color

    uniform_color = args.dataset == 'test'

    batch = next(dl_iter)
    geometries = process_batch()
    print_curr_batch_infos()

    clear_geometry(vis)
    drawn_pcds.clear()
    for g in geometries:
        vis.add_geometry(g)
        drawn_pcds.append(g)


def print_curr_batch_infos():
    global batch
    print('Loaded sample: (scene: {}, sequence: {}, src: {}, target {})'
          .format(batch['label'], batch['seq'], batch['src_id'], batch['target_id']))


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
    transform = torchvision.transforms.Compose(model.get_default_batch_transform())

    def action(vis):
        if vis.scene.has_geometry('src_' + name):
            return
        _batch = copy.deepcopy(batch)
        pred, _ = model(transform(_batch))
        T = np.identity(4)
        T[:3, :4] = pred.detach().cpu().numpy().squeeze()

        add_transformed(vis, T, name, color=color)

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
