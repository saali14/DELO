import torch
from typing import Any, Callable, Optional
from torchmetrics.metric import Metric
from pytorch3d.transforms import matrix_to_euler_angles

class RigidTransformationMetrics(Metric):
    """
    Computes MAE, MSE, RMSE and R2 for rotation in degree and for translation of given transformation matrices.
    Following torchmetrics for R2-score.
    Args:
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

        if prefix is not None:
            self.prefix = prefix + '_'
        else:
            self.prefix = ''

        self.add_state("rot_sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rot_sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("trans_sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("trans_sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # r2 score
        self.add_state("r_sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r_sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("t_sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Input shape: (..., 4, 4) or (..., 3, 4)
        Args:
            preds: Predicted transformations from model
            target: Ground truth transformations
        """
        assert preds.shape == target.shape and preds.size(-1) == 4 and (preds.size(-2) == 3 or preds.size(-2) == 4)

        r_gt_euler_deg = torch.rad2deg(matrix_to_euler_angles(target[..., :3, :3], convention='XYZ'))
        r_pred_euler_deg = torch.rad2deg(matrix_to_euler_angles(preds[..., :3, :3], convention='XYZ'))
        t_gt = target[..., :3, 3]
        t_pred = preds[..., :3, 3]

        n_obs = r_gt_euler_deg.numel()  # (n_samples * 3)
        rot_sum_abs_error = torch.sum(torch.abs(r_pred_euler_deg - r_gt_euler_deg))
        rot_sum_squared_error = torch.sum((r_pred_euler_deg - r_gt_euler_deg) ** 2)
        trans_sum_abs_error = torch.sum(torch.abs(t_pred - t_gt))
        trans_sum_squared_error = torch.sum((t_pred - t_gt) ** 2)

        r_sum_error = torch.sum(r_gt_euler_deg)
        r_sum_squared_error = torch.sum(r_gt_euler_deg ** 2)
        t_sum_error = torch.sum(t_gt)
        t_sum_squared_error = torch.sum(t_gt ** 2)

        self.rot_sum_abs_error += rot_sum_abs_error
        self.rot_sum_squared_error += rot_sum_squared_error
        self.trans_sum_abs_error += trans_sum_abs_error
        self.trans_sum_squared_error += trans_sum_squared_error
        self.total += n_obs

        self.r_sum_error = r_sum_error
        self.r_sum_squared_error = r_sum_squared_error
        self.t_sum_error = t_sum_error
        self.t_sum_squared_error = t_sum_squared_error

    def compute(self):
        """
        Computes mean absolute error over state.
        """

        r_mean_error = self.r_sum_error / self.total
        r_diff = self.r_sum_squared_error - self.r_sum_error * r_mean_error
        r_r2score = 1 - (self.r_sum_squared_error / r_diff)

        t_mean_error = self.t_sum_error / self.total
        t_diff = self.t_sum_squared_error - self.t_sum_error * t_mean_error
        t_r2score = 1 - (self.t_sum_squared_error / t_diff)

        return {
            self.prefix + 'r_mae': self.rot_sum_abs_error / self.total,
            self.prefix + 'r_mse': self.rot_sum_squared_error / self.total,
            self.prefix + 'r_rmse': torch.sqrt(self.rot_sum_squared_error / self.total),
            self.prefix + 'r_r2score': r_r2score,
            self.prefix + 't_mae': self.trans_sum_abs_error / self.total,
            self.prefix + 't_mse': self.trans_sum_squared_error / self.total,
            self.prefix + 't_rmse': torch.sqrt(self.trans_sum_squared_error / self.total),
            self.prefix + 't_r2score': t_r2score,
        }
