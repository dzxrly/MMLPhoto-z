import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from torchvision.transforms import Resize

MAG_PREFIX = "magnitudes"
MAG_DIFF_PREFIX = "magnitudes_diff"


class BuildDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root_dir: str,
        photometric_dir_name: str,
        label_dir_name: str,
        photo_min_max: tuple[float, float],
        target_photo_size: int,
        total_z_categories: int,
        load_mode: str = "train",
        augmentation: bool = False,
        mask_channels: int | None = None,
        mask_prob_in_channel: float | None = None,
        masked_replace: int | float = 0.0,
        eps: float | int = 1e-8,
    ):
        """
        Build a dataloader for the dataset.
        :param dataset_root_dir: dataset root directory
        :param photometric_dir_name: photometric data directory name
        :param label_dir_name : label directory name
        :param photo_min_max: min and max value for photometric data normalization
        :param target_photo_size: target size for photometric data, if not equal to the original size, will be resized
        :param total_z_categories: total number of redshift categories
        :param load_mode: mode to load the dataset, can be "train", "val", or "test"
        :param augmentation: whether to apply data augmentation, default is False
        :param mask_channels: number of channels to mask, default is None, meaning no masking will be applied
        :param mask_prob_in_channel: probability of mask same patch in a channel, default is None, meaning no masking will be applied
        :param masked_replace: value to replace the masked channels, default is 0.0
        :param eps: epsilon value for numerical stability, default is 1e-8
        """
        super().__init__()
        assert load_mode in [
            "train",
            "val",
            "test",
        ], ValueError("load_mode must be in ['train', 'val', 'test']")

        self.photometric_dir = os.path.join(dataset_root_dir, photometric_dir_name)
        self.label_df = pd.read_csv(
            os.path.join(
                dataset_root_dir,
                label_dir_name,
                f"{load_mode}.csv",
            ),
            header=0,
            converters={
                "TARGETID": str,
                "TARGET_RA": str,
                "TARGET_DEC": str,
                "Z": float,
                "lp_zPDF": float,
                "Z_bin_idx": int,
                "lp_zPDF_bin_idx": int,
            },
        )
        self.augmentation = augmentation
        self.mask_channels = mask_channels
        self.mask_prob_in_channel = mask_prob_in_channel
        self.masked_replace = torch.tensor(masked_replace, dtype=torch.float32)
        if self.augmentation:
            print("[INFO] Data augmentation is enabled.")
        self.eps = eps
        self.train = load_mode == "train"
        self.photo_min_max = photo_min_max
        self.target_photo_size = target_photo_size
        self.resize_transform = Resize(
            size=target_photo_size,
            antialias=False,
        )
        self.total_z_categories = total_z_categories

    def __len__(self):
        return len(self.label_df)

    def _load_data(
        self,
        file_path: str,
        allow_pickle: bool = False,
        nan_to_num: bool = False,
    ) -> torch.Tensor:
        # get file extension
        file_extension = os.path.splitext(file_path)[-1].lower()
        assert file_extension in [".npy", ".mat"], RuntimeError(
            f"Unsupported file extension {file_extension}. Only .npy and .mat are supported."
        )
        _data = None
        if file_extension == ".npy":
            _data = np.load(
                os.path.join(file_path),
                allow_pickle=allow_pickle,
            ).astype(np.float32)
            if nan_to_num:
                _data = np.nan_to_num(_data, nan=0, posinf=0, neginf=0)
            _data = torch.tensor(_data, dtype=torch.float32)
        elif file_extension == ".mat":
            _data = joblib.load(
                os.path.join(file_path),
            )
            _data = torch.tensor(_data).float()
            if nan_to_num:
                _data = _data.nan_to_num(nan=0, posinf=0, neginf=0)
        else:
            raise RuntimeError(
                f"Unsupported file extension {file_extension}. Only .npy and .mat are supported."
            )
        # resize
        if _data.shape[-1] != self.target_photo_size:
            _data = self.resize_transform(_data)
        return _data

    def _augmentation(self, photometric: torch.Tensor) -> torch.Tensor:
        device = photometric.device
        c = photometric.shape[0]
        if self.mask_channels is not None and self.mask_channels > 0:
            assert self.mask_channels <= c, RuntimeError(
                f"mask_channels {self.mask_channels} must be less than or equal to the number of channels {c}"
            )
            zero_indices = torch.randperm(c)[: self.mask_channels]
            photometric[zero_indices] = self.masked_replace.to(device=device)
        if self.mask_prob_in_channel is not None and self.mask_prob_in_channel > 0:
            assert 0 < self.mask_prob_in_channel < 1, RuntimeError(
                f"mask_prob_in_channel {self.mask_prob_in_channel} must be in (0, 1)"
            )
            mask_indices = torch.rand_like(photometric) < self.mask_prob_in_channel
            photometric[mask_indices] = self.masked_replace.to(device=device)
        return photometric

    def __getitem__(self, idx):
        data_row = self.label_df.iloc[idx]
        id = data_row["TARGETID"]
        ra = data_row["TARGET_RA"]
        dec = data_row["TARGET_DEC"]
        z = data_row["lp_zPDF"]
        photometric_file_name = data_row["photo_name"]
        z_bin_idx = data_row["Z_bin_idx"]
        assert 0 <= z_bin_idx < self.total_z_categories, ValueError(
            f"z_bin_idx {z_bin_idx} must be in [0, {self.total_z_categories})"
        )
        # one-hot encode z_bin_idx
        z_bin_idx = f.one_hot(
            torch.tensor(z_bin_idx, dtype=torch.long),
            num_classes=self.total_z_categories,
        ).float()
        # Load photometric data
        photometric = self._load_data(
            os.path.join(self.photometric_dir, photometric_file_name),
            nan_to_num=True,
        )
        # min-max normalization
        photometric = (photometric - self.photo_min_max[0]) / (
            self.photo_min_max[1] - self.photo_min_max[0] + self.eps
        )
        if self.augmentation:
            photometric = self._augmentation(photometric)
        # load mag data
        mags = []
        mags_diff = []
        for col in data_row.index:
            if col.startswith(MAG_PREFIX) and not col.startswith(MAG_DIFF_PREFIX):
                mags.append(data_row[col].astype(float))
            if col.startswith(MAG_DIFF_PREFIX):
                mags_diff.append(data_row[col].astype(float))
        mags = torch.tensor(
            mags,
            dtype=torch.float32,
        ).unsqueeze(-1)
        mags_diff = torch.tensor(
            mags_diff,
            dtype=torch.float32,
        ).unsqueeze(-1)
        # min max norm
        mags = (mags - mags.min()) / (mags.max() - mags.min() + self.eps)
        mags_diff = (mags_diff - mags_diff.min()) / (
            mags_diff.max() - mags_diff.min() + self.eps
        )
        label = torch.tensor(z, dtype=torch.float32).unsqueeze(-1)
        return (
            id,
            ra,
            dec,
            photometric,
            mags,
            mags_diff,
            label,
            z_bin_idx,
        )


def build_dataloader(
    config: dict, mode: str, cross_val_name: str = ""
) -> torch.utils.data.DataLoader:
    """
    Build a dataloader for the dataset.
    :param config: configuration dictionary
    :param mode: mode to load the dataset, can be "train", "val", or "test"
    :param cross_val_name: name of the cross-validation, if any
    :return: DataLoader object
    """
    assert mode in ["train", "val", "test"], ValueError(
        "mode must be in ['train', 'val', 'test']"
    )
    # check keys in config
    keys = [
        "eps",
        # dataset settings
        "dataset_root_dir",
        "photometric_dir_name",
        "label_dir_name",
        "augmentation",
        "mask_channels",
        "mask_prob_in_channel",
        "masked_replace",
        "photo_min_max",
        "photo_in_size",
        "z_categories",
        # dataloader settings
        "batch_size",
        "num_workers",
    ]
    for key in keys:
        assert key in config, ValueError(f"{key} not found in config")
    dataset_root_dir = config["dataset_root_dir"]
    label_dir_name = config["label_dir_name"]
    if cross_val_name is not None and cross_val_name != "":
        label_dir_name = os.path.join(dataset_root_dir, label_dir_name, cross_val_name)
    _dataset = BuildDataset(
        dataset_root_dir=dataset_root_dir,
        photometric_dir_name=config["photometric_dir_name"],
        label_dir_name=label_dir_name,
        total_z_categories=config["z_categories"],
        load_mode=mode,
        augmentation=config["augmentation"] and mode == "train",
        mask_channels=config["mask_channels"],
        mask_prob_in_channel=config["mask_prob_in_channel"],
        masked_replace=config["masked_replace"],
        photo_min_max=config["photo_min_max"],
        target_photo_size=config["photo_in_size"],
        eps=config["eps"],
    )
    _dataloader = torch.utils.data.DataLoader(
        dataset=_dataset,
        batch_size=config["batch_size"],
        shuffle=(mode == "train"),
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    return _dataloader
