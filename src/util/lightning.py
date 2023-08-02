import logging
import os
import subprocess

from lightning.pytorch import Trainer
from data.dataloader import add_available_datasets, add_data_specific_args
from model import add_available_models, add_model_specific_args
from util.file_helper import create_dir

def add_parser_arguments(parser, require_checkpoint=False):
    # add trainer specific arguments
    #parser = Trainer.add_argparse_args(parser)
    parser = add_available_models(parser)
    parser = add_available_datasets(parser)

    # parse model and data name
    temp_args, _ = parser.parse_known_args()

    # add model and data specific arguments
    parser = add_model_specific_args(temp_args.model_name, parser)
    parser = add_data_specific_args(temp_args.dataset_type, parser)

    # checkpoint to load weights and hyperparameter
    parser.add_argument('--acceleratorType', type=str, required=True, help='options are cpu, gpu, tpu, cuda, auto ... ')
    parser.add_argument('--checkpoint_path', type=str, metavar='PATH', required=require_checkpoint,  help='Checkpoint to load from.')
    parser.add_argument('--experiment', type=str, help='Experiment name for log path. Default to model_name.')
    parser.add_argument('--disable_save', action='store_true', help='Do not save model to file.)')
    parser.add_argument('--save_weights', action='store_true', help='Only save model weights. Otherwise, save full model.')
    parser.add_argument('--fast_debug', action='store_true', help='Run small fraction of steps and epochs.')
    parser.add_argument('--log_dir', type=str, metavar='PATH', default='../logs', help='Directory to save logs.')
    parser.add_argument('--periodic_save_dir', type=str, metavar='PATH', help='Directory to save logs.')
    parser.add_argument('--anomaly_detection', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=100, help='set maximum number epochs')
    parser.add_argument('--nodes', type=int, default=1, help='no of GPU nodes for distributed learning')
    return parser

def configure_logger(log_path, module='pytorch_lightning', log_level=logging.INFO):
    logger = logging.getLogger(module)
    logger.setLevel(log_level)
    create_dir(log_path)
    file_handler = logging.FileHandler(os.path.join(log_path, "core.log"))
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #log_git_commit(logger)

    # log general info to same file
    core_logger = logging.getLogger('core.log')
    core_logger.setLevel(log_level)
    core_logger.addHandler(file_handler)
    return core_logger

def log_git_commit(logger):
    try:
        commit = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=tformat:%H %ad', '--date=short'], universal_newlines=True).strip().split()
        logger.info('Run on Commit {} from {}.'.format(*commit))
    except subprocess.CalledProcessError:
        pass
