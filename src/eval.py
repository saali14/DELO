import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from callback.log_predictions import LogTestSequence
from callback.loop_closure_eval import LoopClosureEvaluation
from callback.metrics_logger import TestMetricsLogger
from callback.plot_sequence import PlotTestSequence
from callback.recall import RecallCallback
from callback.sequence_metrics import TestSequenceMetricsLogger
from data.dataloader import load_data_module
from model import load_model
from util.lightning import add_parser_arguments, configure_logger


def main(args):
    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None} # remove None entries

    # load model and dataset
    model = load_model(args.model_name, args.checkpoint_path, dict_args)
    dm = load_data_module(args.dataset_type, dict_args)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(args.log_dir, 'test'),
        name=args.experiment if args.experiment else args.model_name,
    )

    test_callbacks = [TestMetricsLogger(tb_logger.log_dir), RecallCallback(tb_logger.log_dir), TestSequenceMetricsLogger(tb_logger.log_dir)]

    if args.dataset_type == 'Kitti':
        test_callbacks.append(PlotTestSequence(tb_logger.log_dir, plot_prediction=True))
    else:
        test_callbacks.append(PlotTestSequence(tb_logger.log_dir, plot_prediction=False))

    if args.model_name == 'dslam' and args.use_descriptor:
        test_callbacks.append(LoopClosureEvaluation(tb_logger.log_dir))

    # configure lightning logging
    logger = configure_logger(tb_logger.log_dir)

    if not args.checkpoint_path:
        logger.warning('No checkpoint given!')
    else:
        logger.info('Loaded model from checkpoint {}'.format(args.checkpoint_path))

    trainer = Trainer.from_argparse_args(args, logger=[tb_logger], callbacks=test_callbacks, limit_test_batches=1.)
    trainer.test(model, datamodule=dm)
    #dm.setup(stage='fit')
    #trainer.test(model, test_dataloaders=dm.val_dataloader())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_parser_arguments(parser, require_checkpoint=False)
    args = parser.parse_args()

    # test
    main(args)
