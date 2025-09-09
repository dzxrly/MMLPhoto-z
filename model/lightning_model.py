import lightning
import torch

from model.loss import (
    GanLoss,
    z_estimate,
    delta_z,
    nmad_z,
    z_bias,
    sigma_n,
    outline_fraction,
)
from model.mml_photo_z import BuildModel


class BuildLightningModel(lightning.LightningModule):

    def __init__(
        self,
        random_seed: int,
        eps: float | int,
        # hyper params
        learn_rate: float,
        cos_annealing_t_0: int,
        cos_annealing_t_mult: int,
        cos_annealing_eta_min: float,
        # photometric & mags settings
        photo_in_channel: int,
        mag_in_size: int,
        extractor_out_dim: int,
        out_channel: int,
        # PyTorch2.0 settings
        enable_torch_2: bool = False,
    ):
        super().__init__()
        print("[INFO] Using Random Seed: ", random_seed)
        self.model = (
            BuildModel(
                img_input_channel=photo_in_channel,
                mags_input_dim=mag_in_size,
                extractor_out_dim=extractor_out_dim,
                out_gaussian_groups=out_channel,
            )
            if not enable_torch_2
            else torch.compile(
                model=BuildModel(
                    img_input_channel=photo_in_channel,
                    mags_input_dim=mag_in_size,
                    extractor_out_dim=extractor_out_dim,
                    out_gaussian_groups=out_channel,
                ),
                backend="inductor",
            )
        )
        if enable_torch_2:
            print("[INFO] Using PyTorch 2.0 compile")
        self.learn_rate = learn_rate
        self.cos_annealing_t_0 = cos_annealing_t_0
        self.cos_annealing_t_mult = cos_annealing_t_mult
        self.cos_annealing_eta_min = cos_annealing_eta_min
        self.estimation_loss = GanLoss(
            out_gaussian_groups=out_channel,
            temperature=0.21,
            gamma=0.13,
            scale_factor=1.06,
            eps=eps,
        )

        self.val_loss = torch.zeros(1).to(self.device)
        self.val_label = []
        self.val_pred = []
        self.val_loss_epoch = 0.0
        self.best_val_loss = 9e10
        self.best_val_mae_epoch = 9e10
        self.best_val_outline_fraction_epoch = 1.0

        self.test_label = []
        self.test_pred = []

        self.step = 0
        self.eps = eps

        self.save_hyperparameters()

    def _log_eval_metrics(
        self,
        prefix: str,
        label: list[torch.Tensor] | torch.Tensor,
        pred: list[torch.Tensor] | torch.Tensor,
    ) -> dict:
        _res_dict = {}
        is_step = isinstance(label, torch.Tensor)
        suffix = "step" if is_step else "epoch"
        if isinstance(pred, list):
            pred = torch.cat(pred, dim=0)
        if isinstance(label, list):
            label = torch.cat(label, dim=0)
        _mae_pred = z_estimate(
            pred=pred,
            mode="mean",
        )
        _mae = torch.mean((_mae_pred - label.squeeze(-1)).abs())
        self.log(
            f"{prefix}_mae_{suffix}",
            _mae,
            prog_bar=True,
            on_step=is_step,
            on_epoch=not is_step,
        )
        _res_dict[f"{prefix}_mae_{suffix}"] = _mae.item()
        _d_z = delta_z(
            pred=pred,
            label=label,
            mode="mean",
        )
        metrics = [
            ("nmad_z", nmad_z(_d_z, factor=1.4826)),
            ("z_bias", z_bias(_d_z)),
            ("sigma_0.05", sigma_n(_d_z, n=0.05)),
            ("sigma_0.15", sigma_n(_d_z, n=0.15)),
            ("outline_fraction", outline_fraction(_d_z, threshold=0.1)),
        ]
        for metric_name, metric_value in metrics:
            key = f"{prefix}_{metric_name}_{suffix}"
            self.log(
                key,
                metric_value,
                prog_bar=True,
                on_step=is_step,
                on_epoch=not is_step,
            )
            _res_dict[key] = metric_value
        return _res_dict

    def _get_estimation_loss(
        self,
        mags2mags_judge: torch.Tensor,
        img2mags_judge: torch.Tensor,
        mags2img_judge: torch.Tensor,
        img2img_judge: torch.Tensor,
        label: torch.Tensor,
        z_bin_idx: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        mode: str = "train",
    ):
        (
            _loss,
            _c1_loss,
            _c2_loss,
            _c3_loss,
            _c4_loss,
            _contrastive_loss,
            _z1_loss,
            _z2_loss,
            _z3_loss,
            _z4_loss,
        ) = self.estimation_loss(
            mags2mags_judge,
            img2mags_judge,
            mags2img_judge,
            img2img_judge,
            label,
            z_bin_idx,
            z1,
            z2,
            z3,
            z4,
            mode,
        )
        self.log(
            f"{mode}_c1_loss",
            _c1_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_c2_loss",
            _c2_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_c3_loss",
            _c3_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_c4_loss",
            _c4_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_contrastive_loss",
            _contrastive_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_z1_loss",
            _z1_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_z2_loss",
            _z2_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_z3_loss",
            _z3_loss,
            prog_bar=True,
            on_step=True,
        )
        self.log(
            f"{mode}_z4_loss",
            _z4_loss,
            prog_bar=True,
            on_step=True,
        )
        return _loss

    def training_step(self, batch, batch_idx):
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            mags_diff,
            label,
            z_bin_idx,
        ) = batch
        (
            z1,
            z2,
            z3,
            z4,
            img_feature,
            mags_feature,
            img2mags_feature,
            mags2img_feature,
            img2img_judge,
            mags2img_judge,
            img2mags_judge,
            mags2mags_judge,
        ) = self.model(mags, mags_diff, photometric)
        _estimation_loss = self._get_estimation_loss(
            mags2mags_judge=mags2mags_judge,
            img2mags_judge=img2mags_judge,
            mags2img_judge=mags2img_judge,
            img2img_judge=img2img_judge,
            label=label,
            z_bin_idx=z_bin_idx,
            z1=z1,
            z2=z2,
            z3=z3,
            z4=z4,
            mode="train",
        )
        self.log(
            "train_estimation_loss",
            _estimation_loss,
            prog_bar=True,
            on_step=True,
        )
        self._log_eval_metrics(
            "train",
            label=label.detach().cpu(),
            pred=z1.detach().cpu(),
        )
        return _estimation_loss

    def on_validation_epoch_start(self):
        self.val_loss = torch.zeros(1).to(self.device)
        self.val_loss_epoch = 0
        self.step = 0
        self.val_label = []
        self.val_pred = []

    def validation_step(self, batch, batch_idx):
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            mags_diff,
            label,
            z_bin_idx,
        ) = batch
        (
            z1,
            z2,
            z3,
            z4,
            img_feature,
            mags_feature,
            img2mags_feature,
            mags2img_feature,
            img2img_judge,
            mags2img_judge,
            img2mags_judge,
            mags2mags_judge,
        ) = self.model(mags, mags_diff, photometric)
        _estimation_loss = self._get_estimation_loss(
            mags2mags_judge=mags2mags_judge,
            img2mags_judge=img2mags_judge,
            mags2img_judge=mags2img_judge,
            img2img_judge=img2img_judge,
            label=label,
            z_bin_idx=z_bin_idx,
            z1=z1,
            z2=z2,
            z3=z3,
            z4=z4,
            mode="eval",
        )
        self.val_loss_epoch += _estimation_loss
        self.step += 1
        self.val_label.append(label.detach().cpu())
        self.val_pred.append(z1.detach().cpu())
        return _estimation_loss

    def on_validation_epoch_end(self):
        _val_loss = self.val_loss_epoch / self.step
        self.log(
            "val_loss_epoch",
            _val_loss,
            prog_bar=True,
            on_epoch=True,
        )
        if _val_loss < self.best_val_loss:
            self.best_val_loss = _val_loss
            self.log(
                "best_val_loss",
                self.best_val_loss,
                prog_bar=True,
                on_epoch=True,
            )

        val_metrics = self._log_eval_metrics(
            prefix="val",
            label=self.val_label,
            pred=self.val_pred,
        )
        if val_metrics["val_mae_epoch"] < self.best_val_mae_epoch:
            self.best_val_mae_epoch = val_metrics["val_mae_epoch"]
            self.log(
                "best_val_mae_epoch",
                self.best_val_mae_epoch,
                prog_bar=True,
                on_epoch=True,
            )
        if (
            val_metrics["val_outline_fraction_epoch"]
            < self.best_val_outline_fraction_epoch
        ):
            self.best_val_outline_fraction_epoch = val_metrics[
                "val_outline_fraction_epoch"
            ]
            self.log(
                "best_val_outline_fraction_epoch",
                self.best_val_outline_fraction_epoch,
                prog_bar=True,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        (
            id,
            ra,
            dec,
            photometric,
            mags,
            mags_diff,
            label,
            z_bin_idx,
        ) = batch
        (
            z1,
            z2,
            z3,
            z4,
            img_feature,
            mags_feature,
            img2mags_feature,
            mags2img_feature,
            img2img_judge,
            mags2img_judge,
            img2mags_judge,
            mags2mags_judge,
        ) = self.model(mags, mags_diff, photometric)
        _estimation_loss = self._get_estimation_loss(
            mags2mags_judge=mags2mags_judge,
            img2mags_judge=img2mags_judge,
            mags2img_judge=mags2img_judge,
            img2img_judge=img2img_judge,
            label=label,
            z_bin_idx=z_bin_idx,
            z1=z1,
            z2=z2,
            z3=z3,
            z4=z4,
            mode="eval",
        )
        self.test_label.append(label.detach().cpu())
        self.test_pred.append(z1.detach().cpu())
        return _estimation_loss

    def on_test_epoch_end(self):
        self._log_eval_metrics(
            prefix="test",
            label=self.test_label,
            pred=self.test_pred,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            lr=self.learn_rate,
            params=self.model.parameters(),
            eps=self.eps,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cos_annealing_t_0,
            T_mult=self.cos_annealing_t_mult,
            eta_min=self.cos_annealing_eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "name": "lr"},
        }
