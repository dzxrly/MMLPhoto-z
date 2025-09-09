import torch
from torchinfo import summary

from config.config import config
from model.mml_photo_z import BuildModel

torch.random.manual_seed(42)

if __name__ == "__main__":
    device = torch.device("cpu")

    model = BuildModel(
        img_input_channel=config["photo_in_channel"],
        mags_input_dim=config["mag_in_size"],
        extractor_out_dim=config["extractor_out_dim"],
        out_gaussian_groups=config["out_gaussian_groups"],
    )
    model.to(device)
    summary(
        model,
        input_size=[
            (
                config["batch_size"],
                config["mag_in_size"] // 2 + 1,
                1,
            ),
            (
                config["batch_size"],
                config["mag_in_size"] // 2,
                1,
            ),
            (
                config["batch_size"],
                config["photo_in_channel"],
                config["photo_in_size"],
                config["photo_in_size"],
            ),
        ],
        mode="train",
        depth=10,
        device=device,
    )
