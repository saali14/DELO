import os
import logging
import torch

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme 

from util.file_helper import create_dir
from util.timer import Timer

myProgressBar = RichProgressBar(
        leave=True,
        theme=RichProgressBarTheme(
            description="green_yellow", 
            progress_bar="green1", 
            progress_bar_finished="green1",
            progress_bar_pulse="yellow2", 
            batch_progress="green_yellow", 
            time="pink3", 
            processing_speed="orange1", 
            metrics="orange1",
            )
        )

class ValidationMetricsLogger(Callback):
    def __init__(self, log_path):
        self.logger = _configure_logger(log_path, 'val')
        self.logger.propagate = False

    def on_validation_end(self, trainer, pl_module):
        metrics = pl_module.metrics.compute()
        loss = pl_module.val_loss.compute()
        #pl_module.val_loss.reset()
        metrics['val_loss'] = loss
        metrics['step'] = pl_module.global_step

        self.logger.info(format_dict(metrics))

class TestMetricsLogger(Callback):
    def __init__(self, log_path):
        self.logger = _configure_logger(log_path, 'test')
        self.batch_time = Timer()
        self.test_time = Timer()
        self.sample_count = 0

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_time.tic()


    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.batch_time.tic()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_time.toc()
        if isinstance(batch['idx'], list):
            self.sample_count += len(batch['idx'])
        else:
            self.sample_count += batch['idx'].size(0)

    def on_test_end(self, trainer, pl_module):
        metrics = pl_module.metrics.compute()
        metrics['forward_time'] = pl_module.forward_time.sum / max(self.sample_count, 1)
        metrics['forward_batch_time'] = pl_module.forward_time.avg
        metrics['total_time'] = self.batch_time.sum / max(self.sample_count, 1)
        metrics['avg_total_time'] = self.test_time.toc() / max(self.sample_count, 1)
        self.logger.info(format_dict(metrics))

class TrainMetricsLogger(Callback):
    def __init__(self, log_path):
        self.logger = _configure_logger(log_path, 'train', format='%(message)s')
        self.logger.propagate = False

    def on_train_epoch_end(self, trainer, pl_module, unused = None):
        metrics = pl_module.train_metrics.compute()
        loss = pl_module.train_loss.compute()
        metrics['train_loss'] = loss
        metrics['epoch'] = trainer.current_epoch
        self.logger.info(format_dict2(metrics, 'train'))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = pl_module.metrics.compute()
        loss = pl_module.val_loss.compute()
        metrics['val_loss'] = loss
        metrics['epoch'] = trainer.current_epoch
        self.logger.info(format_dict2(metrics, 'val'))

def format_dict(dict):
    return {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}

def format_dict2(dict, stage):
    d = {k.replace(stage+'_', ''): (v.item() if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}
    return 'A->B:: Stage: %s, Epoch: %d, Loss: %f, Rot_MSE: %f, Rot_RMSE: %f, Rot_MAE: %f, Rot_R2: %f, Trans_MSE: %f, Trans_RMSE: %f, Trans_MAE: %f, Trans_R2: %f' % (stage, d['epoch'], d['loss'], d['r_mse'], d['r_rmse'], d['r_mae'], d['r_r2score'], d['t_mse'], d['t_rmse'], d['t_mae'], d['t_r2score'])

def _configure_logger(path, subset, format='%(asctime)s %(message)s'):
    if not os.path.exists(path):
        create_dir(path)
    logger = logging.getLogger(subset + '_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(path, subset + '.log'))
    formatter = logging.Formatter(fmt=format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
