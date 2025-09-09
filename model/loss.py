import math

import torch
import torch.nn as nn
import torch.nn.functional as f


def z_estimate(
    pred: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Calculate the delta_z from the prediction.
    :param pred: predictions, shape like (batch_size, N*3)
    :param mode: Gaussian mixture mode, can be 'mean' or 'max', 'mean' will return the weighted mean of the
        Gaussian components, 'max' will return the component with the highest weight
    :return: z estimate, shape like (batch_size, 1)
    """
    mean = pred[:, :5]  # (batch_size, N)
    weight = pred[:, 10:15]  # (batch_size, N)
    if mode == "mean":
        x = torch.sum(mean * weight, dim=-1)  # (batch_size,)
        return x
    elif mode == "max":
        max_index = torch.argmax(weight, dim=-1)  # (batch_size,)
        x = mean[torch.arange(mean.size(0)), max_index]  # (batch_size,)
        return x
    else:
        raise ValueError("mode must be 'mean' or 'max', got {}".format(mode))


def delta_z(
    pred: torch.Tensor,
    label: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Calculate the delta_z from the prediction.
    :param pred: predictions, shape like (batch_size, N*3)
    :param label: labels, actual redshift, shape like (batch_size, 1)
    :param mode: Gaussian mixture mode, can be 'mean' or 'max', 'mean' will return the weighted mean of the
        Gaussian components, 'max' will return the component with the highest weight
    :return: delta_z, shape like (batch_size, 1)
    """
    pred_z = z_estimate(
        pred=pred,
        mode=mode,
    )
    pred_z = pred_z.unsqueeze(-1) if pred_z.dim() == 1 else pred_z
    label = label.unsqueeze(-1) if label.dim() == 1 else label
    return (pred_z - label) / (1 + label)


def z_bias(
    d_z: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the bias of the redshift estimation.
    :param d_z: predictions, shape like (all,1)
    :return: Mean(delta_z), shape like (1)
    """
    return d_z.mean()


def nmad_z(
    d_z: torch.Tensor,
    factor: float = 1.4826,
) -> torch.Tensor:
    """
    Calculate the Normalized Median Absolute Deviation (NMAD) of the redshift estimation.
    :param d_z: predictions, shape like (all,1)
    :param factor: factor to multiply with the median absolute deviation, default is 1.48
    :return: NMAD, shape like (1)
    """
    median_d_z = torch.median(d_z)
    return factor * torch.median(torch.abs(d_z - median_d_z))


def sigma_n(
    d_z: torch.Tensor,
    n: float = 0.05,
) -> torch.Tensor:
    """
    Sigma_n metric
    :param d_z: predictions, shape like (all,1)
    :param n: threshold for the sigma_n metric, default is 0.05
    :return: Sigma_n, shape like (1), bigger is better
    """
    d_z = d_z.abs()
    mask = n > d_z
    return torch.sum(mask) / len(d_z)


def outline_fraction(
    d_z: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Outline fraction metric
    :param d_z: predictions, shape like (all,1)
    :param threshold: threshold for the outline fraction metric, default is 0.1
    :return: Outline fraction, shape like (1), smaller is better
    """
    d_z = d_z.abs()
    mask = d_z >= threshold
    return torch.sum(mask) / len(d_z)


class CRPSLoss(nn.Module):

    def __init__(self, out_gaussian_groups: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.out_gaussian_groups = out_gaussian_groups
        self.sdf = torch.distributions.Normal(0.0, 1.0)

    def crps_normal(
        self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if y.shape[-1] == 1:
            y = y.expand(-1, self.out_gaussian_groups)
        x = (y - mu) / (sigma + self.eps)
        cdf = self.sdf.cdf(x)
        pdf = torch.exp(self.sdf.log_prob(x))
        return sigma * (x * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))

    def forward(
        self, pred: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = pred[:, :5] + self.eps
        std = pred[:, 5:10] + self.eps
        weight_logits = f.softmax(pred[:, 10:15], dim=-1)
        # Term 1: sum_i w_i * CRPS(mu_i, sigma_i; y)
        component_crps = self.crps_normal(mean, std, label)
        term1 = torch.sum(weight_logits * component_crps, dim=1)
        # Term 2: 0.5 * sum_{i,j} w_i w_j * CRPS(mu_i - mu_j, sqrt(sigma_i^2 + sigma_j^2); 0)
        mu_i = mean.unsqueeze(2)
        mu_j = mean.unsqueeze(1)
        std_i = std.unsqueeze(2)
        std_j = std.unsqueeze(1)
        mu_diff = mu_i - mu_j
        std_comb = torch.sqrt(std_i**2 + std_j**2 + self.eps)
        crps_cross = self.crps_normal(mu_diff, std_comb, torch.zeros_like(mu_diff))
        weight_i = weight_logits.unsqueeze(2)
        weight_j = weight_logits.unsqueeze(1)
        weight_pair = weight_i * weight_j
        term2 = 0.5 * torch.sum(weight_pair * crps_cross, dim=(1, 2))

        loss = term1 - term2
        return loss.mean(), weight_logits, std


class ContrastiveLoss(nn.Module):

    def __init__(
        self,
        temperature: float = 0.21,
        gamma: float = 0.13,
        scale_factor: float = 1.06,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.scale_factor = scale_factor
        self.eps = eps

    def forward(
        self,
        mags_feature: torch.Tensor,
        img_feature: torch.Tensor,
        z_category_label: torch.Tensor,
    ) -> torch.Tensor:
        mags_feature = f.normalize(mags_feature, dim=1)
        img_feature = f.normalize(img_feature, dim=1)
        sim_mags2img = torch.matmul(mags_feature, img_feature.T) / self.temperature
        sim_mags2mags = torch.matmul(mags_feature, mags_feature.T) / self.temperature
        sim_img2img = torch.matmul(img_feature, img_feature.T) / self.temperature
        label_sim = torch.matmul(z_category_label, z_category_label.T).clamp(max=1.0)
        pro_inter = label_sim / label_sim.sum(dim=1, keepdim=True).clamp(min=self.eps)
        label_sim_intra = (
            label_sim - torch.eye(label_sim.shape[0], device=label_sim.device)
        ).clamp(min=0)
        pro_intra = label_sim_intra / label_sim_intra.sum(dim=1, keepdim=True).clamp(
            min=self.eps
        )

        logits_mags2img = sim_mags2img - torch.log(
            torch.exp(self.scale_factor * sim_mags2img).sum(dim=1, keepdim=True)
        )
        logits_img2mags = sim_mags2img.T - torch.log(
            torch.exp(self.scale_factor * sim_mags2img.T).sum(dim=1, keepdim=True)
        )
        logits_mags2mags = sim_mags2mags - torch.log(
            torch.exp(self.scale_factor * sim_mags2mags).sum(dim=1, keepdim=True)
        )
        logits_img2img = sim_img2img - torch.log(
            torch.exp(self.scale_factor * sim_img2img).sum(dim=1, keepdim=True)
        )

        mean_log_prob_pos_mags2img = (pro_inter * logits_mags2img).sum(dim=1)
        mean_log_prob_pos_img2mags = (pro_inter * logits_img2mags).sum(dim=1)
        mean_log_prob_pos_mags2mags = (pro_intra * logits_mags2mags).sum(dim=1)
        mean_log_prob_pos_img2img = (pro_intra * logits_img2img).sum(dim=1)

        return (
            -mean_log_prob_pos_mags2img.mean()
            - mean_log_prob_pos_img2mags.mean()
            - self.gamma
            * (mean_log_prob_pos_mags2mags.mean() + mean_log_prob_pos_img2img.mean())
        )


class GanLoss(nn.Module):

    def __init__(
        self,
        out_gaussian_groups: int,
        temperature: float = 0.21,
        gamma: float = 0.13,
        scale_factor: float = 1.06,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.crps_loss = CRPSLoss(
            out_gaussian_groups=out_gaussian_groups,
            eps=eps,
        )
        self.contrastive_loss = ContrastiveLoss(
            temperature=temperature,
            gamma=gamma,
            scale_factor=scale_factor,
            eps=eps,
        )

    def forward(
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
    ) -> tuple:
        assert mode in ["train", "eval"], NotImplementedError(
            f"Mode {mode} not implemented, must be in ['train', 'eval']"
        )
        b = label.shape[0]
        mags_md = torch.ones(b, dtype=torch.long, device=label.device)
        img_md = torch.zeros(b, dtype=torch.long, device=label.device)

        c1 = self.cross_entropy_loss(mags2mags_judge, mags_md)
        c2 = self.cross_entropy_loss(img2mags_judge, img_md)
        c3 = self.cross_entropy_loss(mags2img_judge, img_md)
        c4 = self.cross_entropy_loss(img2img_judge, img_md)

        z1_loss, _, _ = self.crps_loss(z1, label)
        z2_loss, _, _ = self.crps_loss(z2, label)
        z3_loss, _, _ = self.crps_loss(z3, label)
        z4_loss, _, _ = self.crps_loss(z4, label)

        contrastive_loss = self.contrastive_loss(
            mags_feature=mags2mags_judge,
            img_feature=img2img_judge,
            z_category_label=z_bin_idx,
        )

        return (
            (
                5 * z1_loss + 0.4 * (c1 + c2 + c3 + c4) + 0.3 * contrastive_loss
                if mode == "train"
                else z1_loss
            ),
            c1,
            c2,
            c3,
            c4,
            contrastive_loss,
            z1_loss,
            z2_loss,
            z3_loss,
            z4_loss,
        )
