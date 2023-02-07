from functools import lru_cache
import os
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from rich.console import Console

# from seqvec.analysis.metrics import concordance_index_compute
from torch import Tensor, nn
from lifelines.utils import concordance_index
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)
import wandb
from aika.visual.plots import scatter_plot
import math
console = Console()

def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)
    return lr_lambda


class PDBModelModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        optimizer_hparams: Union[dict, None] = None,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        batch_size: int = 32,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        num_workers: Union[int, None] = None,
        weight_decay=1e-2,
        scheduler_name: str = "ReduceLROnPlateau",
        scheduler_monitor: str = "loss",
        decay_milestone: Union[None, List[int]] = None,
        **kwargs,
    ) -> None:

        super().__init__()
        # self.save_hyperparameters(logger=False)
        decay_milestone = decay_milestone or [50, 80, 110, 150, 200, 220, 250]
        # self.lr: float = lr
        self.scheduler_monitor: str = scheduler_monitor
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.num_workers: Union[int, None] = num_workers or os.cpu_count()
        self.model = model
        # self.loss_module = nn.MSELoss()
        self.mse: MeanSquaredError = MeanSquaredError()
        self.rmse: MeanSquaredError = MeanSquaredError(squared=False)
        self.r2: R2Score = R2Score()
        self.mae: MeanAbsoluteError = MeanAbsoluteError()
        self.spearmanr: SpearmanCorrCoef = SpearmanCorrCoef()
        self.pearsonr: SpearmanCorrCoef = PearsonCorrCoef()
        # self.ev: ExplainedVariance = ExplainedVariance()
        self.softmax = nn.Softmax(dim=0)
        # self.CustomCI = concordance_index_compute
        self.result_dict_train = {}
        self.result_dict_test = {}
        self.result_dict_valid = {}
        self.schedulers_name = scheduler_name
        self.save_hyperparameters(logger=False)
        self.alpha_custom = 0.7

    def configure_optimizers(self) -> Tuple[List, List]:
        #  support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                **self.hparams.optimizer_hparams,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                **self.hparams.optimizer_hparams,
            )
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        if self.schedulers_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.hparams.decay_milestone, gamma=0.5
            )
        elif self.schedulers_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=0, last_epoch=-1
            )
        elif self.schedulers_name == "ReduceLROnPlateau":
            # decay lr if no improvement in loss
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.8,
                patience=3,
                verbose=True,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
            )
        elif self.schedulers_name == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, cycle_momentum=True)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss",
                "interval": "epoch",
                "cycle_momentum": True,
                "frequency": 3  # "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        # return [optimizer], [scheduler]
        
    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs


    def forward(self, x) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> dict:
        x, y_true = batch
        y_pred = self.all_prediction(x)
        # loss = self.rmse(y_pred, y_true)
        loss = self.mse(y_pred, y_true)
        # console.print(y_pred,"@@@@@@@@@@@")
        # R = self.pearsonr(y_pred, y_true)
        # loss = self.alpha_custom*(1-R)+(1-self.alpha_custom)*loss
        #mae loss
        # loss = self.mae(y_pred, y_true)
        
        self.log("loss", loss)
        tensorboard_logs: dict[str, Tensor] = {"train_rmse_loss": loss}
        progress_bar_metrics: dict[str, Tensor] = tensorboard_logs
        # return {"loss": loss,
        #         "log": progress_bar_metrics,
        #         "progress_bar": progress_bar_metrics}
        return {
            "loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y_true = batch
        # x = x.view(x.size(0), -1)
        y_pred = self.all_prediction(x)
        # loss = self.rmse(y_pred, y_true)
        # R = self.pearsonr(y_pred, y_true)
        # loss = self.alpha_custom*(1-R)+(1-self.alpha_custom)*loss
        loss = self.mse(y_pred, y_true)
        # loss = self.mae(y_pred,y_true)
        self.log("val_loss", loss)
        return {
            "val_loss": loss,
            "pred": y_pred,
            "true": y_true,
        }
        # return [y_pred, y_true]

    def validation_epoch_end(self, outputs) -> None:
        _y_pred, _y_true = self.metric_and_log(outputs, title="val", log_plot=True)
        # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # self.log("val_loss", avg_loss)
        # return {"val_loss": avg_loss}

    def all_prediction(self, x):
        result = self.model(x)
        return result.flatten()

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y_true = batch
        y_pred = self.model(x)
        y_pred: torch.Tensor = torch.flatten(y_pred)
        # loss = self.rmse(y_true, y_pred)
        # loss = self.mae(y_true, y_pred)
        loss = self.mse(y_pred, y_true)
        return {
            "loss": loss,
            "pred": y_pred,
            "true": y_true,
        }

    def test_epoch_end(self, outputs) -> None:
        _y_pred, _y_true = self.metric_and_log(outputs, title="test", log_plot=True)

    def compute_metrics(self, y_pred, y_true):
        metric_mse = self.mse(y_pred, y_true)
        metric_mae = self.mae(y_pred, y_true)
        metric_rmse = self.rmse(y_pred, y_true)
        metric_r2 = self.r2(y_pred, y_true)
        metric_spear = self.spearmanr(y_pred, y_true)
        metric_pearsonr = self.pearsonr(y_pred, y_true)
        # concordnace_index = concordance_index_compute(y_pred, y_true)
        concordnace_index = concordance_index(
            y_true.cpu().numpy(), y_pred.cpu().numpy()
        )
        return (
            metric_mse,
            metric_mae,
            metric_rmse,
            metric_r2,
            metric_spear,
            metric_pearsonr,
            concordnace_index,
        )

    def metric_and_log(self, outputs, title, log_plot=True):
        # y_pred = [x["pred"] for x in outputs if x["pred"] is not None]
        # y_true = [x["true"] for x in outputs if x["true"] is not None]
        # y_pred = [x["pred"] for x in outputs if x.get("pred")]
        # y_true = [x["true"] for x in outputs if x.get("true")]
        y_pred, y_true = zip(
            *[
                (x["pred"], x["true"])
                for x in outputs
                if all(k in x for k in ("pred", "true"))
            ]
        )
        y_pred = torch.cat(y_pred, dim=0).detach()
        y_true = torch.cat(y_true, dim=0).detach()
        (
            _mse,
            _mae,
            _rmse,
            _r2,
            _spear,
            _pearsonr,
            _ci,
        ) = self.compute_metrics(y_pred, y_true)
        self.log_dict(
            {
                f"{title}_mse": _mse,
                f"{title}_rmse": _rmse,
                f"{title}_r2": _r2,
                f"{title}_spear": _spear,
                f"{title}_ci": _ci,
                f"{title}_mae": _mae,
                f"{title}_pearsonr": _pearsonr,
            },
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # self.log(f"{id}_rmse", _rmse)
        # self.log(f"{id}_mse", _mse)
        # self.log(f"{id}_r2", _r2)
        # self.log(f"{id}_spear", _spear)
        # self.log(f"{id}_ci", _ci, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"{id}_mae", _mae)
        # self.log(f"{id}_pearsonr", _pearsonr)

        if log_plot:
            fig = scatter_plot(
                y_pred.cpu().numpy(),
                y_true.cpu().numpy(),
                title=f"{title} scatter plot",
            )
            wandb.log({f"{title}_scatter_plot": wandb.Image(fig)})
            fig.clf()
        return y_pred, y_true
