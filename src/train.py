import torch
import logging
import os
import subprocess

from argparse import ArgumentParser as argparser

#from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, ArgsType

from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from callback.metrics_logger import myProgressBar, ValidationMetricsLogger, TrainMetricsLogger
from callback.save_periodic import SavePeriodicCheckpoint
from util.lightning import add_parser_arguments, configure_logger
from data.dataloader import load_data_module
from model import load_model



def main(args):
    print("Partial Optimal Transport based Deep Evidential Lidar Odometry (POT-DELO).., Sk Aziz Ali, DFKI GmbH, University of Luxembourg")
    print(torch.cuda.device_count())
    expname = args.experiment if args.experiment else args.model_name
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(args.log_dir, 'train'), name=expname)

    # configure lightning logging
    logger = configure_logger(tb_logger.log_dir)
    dict_args = {k: v for k, v in vars(args).items() if v is not None}  # remove None entries

    # load model and dataset
    model = load_model(args.model_name, args.checkpoint_path, dict_args)
    data_module = load_data_module(args.dataset_type, dict_args)

    if args.checkpoint_path:
        logger.info('model loaded from checkpoint {}'.format(args.checkpoint_path))
    callbacks = []

    if not args.disable_save:
        print("monitoring")
        checkpoint_callback = ModelCheckpoint(
            monitor='train_total_loss',
            save_top_k=1,
            mode='min',
            save_last=True,
            dirpath=os.path.join(tb_logger.log_dir, 'checkpoint'),
            save_weights_only=args.save_weights,
        )
        callbacks.append(checkpoint_callback)
    if args.periodic_save_dir is not None:
        checkpoint_epochs_callback = SavePeriodicCheckpoint(dirpath=os.path.join(args.periodic_save_dir, tb_logger.name, 'version_' + str(tb_logger.version)),)
        callbacks.append(checkpoint_epochs_callback)
    callbacks.append(myProgressBar)
    callbacks.append(TrainMetricsLogger(tb_logger.log_dir))
    callbacks.append(ValidationMetricsLogger(tb_logger.log_dir))
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    trainer = Trainer(accelerator=args.acceleratorType,
                      devices='auto',
                      num_nodes=args.nodes,
                      max_epochs=args.max_epochs,
                      log_every_n_steps=2,
                      limit_train_batches=32,
                      limit_test_batches=32,
                      callbacks=callbacks,
                      logger=tb_logger,
                      enable_checkpointing=not args.disable_save,
                      detect_anomaly=args.anomaly_detection)

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparser()
    parser = add_parser_arguments(parser)
    args = parser.parse_args()
    main(args)

