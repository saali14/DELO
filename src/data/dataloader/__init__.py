from data.dataloader.kittiDataModule import KittiDataModule

dataModules = {
    'Kitti-360': KittiDataModule,
    'Kitti': KittiDataModule,
    'waymo': KittiDataModule,
    }

def load_data_module(dataset, kwargs):
    d = dataModules[dataset]
    if d is None:
        raise NotImplementedError

    return d(**kwargs)

def add_data_specific_args(dataset, parser):
    d = dataModules[dataset]
    if d is None:
        raise NotImplementedError
    return d.add_data_specific_args(parser)

def add_available_datasets(parser):
    parser.add_argument('--dataset_type', default='Kitti', choices=dataModules.keys(), metavar='DATASET', help='dataset type')
    return parser
