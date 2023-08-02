import torch

from typing import Any, Callable, Optional
from torchmetrics.metric import Metric


class EvidenceMetric(Metric):
    """
    Computes the mean over a sequence of values
    Args:
        reset_on_compute:
            Calls ``reset()`` after result is computed.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """

    def __init__(
        self,
        reset_on_compute: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        prefix: str = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.reset_on_compute = reset_on_compute

        if prefix is not None:
            self.prefix = prefix + '_'
        else:
            self.prefix = ''

        self.add_state("rot_prediction_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rot_aleatoric_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rot_epistemic_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("trans_prediction_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("trans_aleatoric_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("trans_epistemic_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, evidential_output: torch.Tensor):
        """
        Update state.
        Args:
            x: Single value or sequence of values
        """
        gamma, nu, alpha, beta = evidential_output
        self.rot_prediction_sum += torch.sum(gamma[:, :3])
        rot_aleatoric = beta[:, :3] / (alpha[:, :3] - 1)
        self.rot_aleatoric_sum += torch.sum(rot_aleatoric)
        self.rot_epistemic_sum += torch.sum(rot_aleatoric / nu[:, :3])
        self.trans_prediction_sum += torch.sum(gamma[:, 3:])
        trans_aleatoric = beta[:, 3:] / (alpha[:, 3:] - 1)
        self.trans_aleatoric_sum += torch.sum(trans_aleatoric)
        self.trans_epistemic_sum += torch.sum(trans_aleatoric / nu[:, 3:])
        self.total += gamma.size(0) * 3

    def compute(self):
        return {
            self.prefix + 'rot_prediction': self.rot_prediction_sum / self.total,
            self.prefix + 'rot_aleatoric': self.rot_aleatoric_sum / self.total,
            self.prefix + 'rot_epistemic': self.rot_epistemic_sum / self.total,
            self.prefix + 'trans_prediction': self.trans_prediction_sum / self.total,
            self.prefix + 'trans_aleatoric': self.trans_aleatoric_sum / self.total,
            self.prefix + 'trans_epistemic': self.trans_epistemic_sum / self.total,
        }
