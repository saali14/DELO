import os
import open3d as o3d
from argparse import ArgumentParser

import torchvision

from model import load_model
from util.lightning import add_parser_arguments
from util.file_helper import create_dir

import numpy as np


def main(args):
    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None}  # remove None entries

    # load model and dataset
    model = load_model(args.model_name, args.checkpoint_path, dict_args)

    src_pcd = o3d.io.read_point_cloud(args.src)
    tgt_pcd = o3d.io.read_point_cloud(args.target)
    src = np.asarray(src_pcd.points).astype('float32')
    tgt = np.asarray(tgt_pcd.points).astype('float32')
    print(src.shape)

    sample = {'points_target': tgt, 'points_src': src, 'idx': 0}

    transform = torchvision.transforms.Compose(model.get_default_batch_transform())

    model.to('cuda')
    pred, _ = model.predict_step(transform(sample), batch_idx=0)
    T = np.identity(4)
    T[:3, :4] = pred.detach().cpu().numpy().squeeze()

    src_t = src_pcd.transform(T)

    np.savetxt(os.path.join(args.out, 'pred.log'), T)
    create_dir(args.out)
    o3d.io.write_point_cloud(os.path.join(args.out, 'src_t.ply'), src_t, write_ascii=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_parser_arguments(parser, require_checkpoint=True)
    parser.add_argument('--src', type=str, metavar='PATH', required=True)
    parser.add_argument('--target', type=str, metavar='PATH', required=True)
    parser.add_argument('--out', type=str, metavar='PATH', required=True)
    args = parser.parse_args()
    main(args)
