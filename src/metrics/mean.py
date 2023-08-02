import torch
from typing import Any, Callable, Optional
from torchmetrics.metric import Metric


class MeanValue(Metric):
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
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.reset_on_compute = reset_on_compute

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: torch.Tensor):
        """
        Update state.
        Args:
            x: Single value or sequence of values
        """
        self.sum += torch.sum(x)
        self.total += x.numel()

    def compute(self):
        """
        Computes mean absolute error over state.
        """

        mean = (self.sum / self.total).clone()
        if self.reset_on_compute:
            self.reset()
        return mean
