import argparse
import json
import os
import shutil

import pandas as pd
import torch
from tqdm.rich import tqdm

from config.config import config
from dataloader.dataloader import build_dataloader
from model.lightning_model import BuildLightningModel
from model.loss import delta_z, nmad_z, sigma_n, z_bias, z_estimate, outline_fraction

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/2025-07-21_10-37-52/checkpoints/best-epoch=2-val_loss_epoch=1.85020.ckpt"
CROSS_FOLD = "fold_0"
RES_SAVE_DIR = "./results"
PDF_MODE = "mean"


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    try:
        model = BuildLightningModel.load_from_checkpoint(
            os.path.join(model_path),
            map_location=device,
        ).model
        # remove model. prefix
        state_dict = {k.replace("model.", ""): v for k, v in model.state_dict().items()}
        model.load_state_dict(state_dict)
        print("[Info] Load model from {}".format(model_path))
    except Exception as e:
        print("[Error] Load model from {} failed".format(model_path))
        raise e
    return model


def inference(
    settings: dict,
    output_mode: str,
    ckpt_path: str,
    dataset_fold: str,
    res_save_dir: str,
    clear_res_dir: bool = False,
    device: torch.device = torch.device("cpu"),
    cuda_infer_timer: bool = True,
) -> None:
    assert output_mode in ["both", "pdf", "point"], ValueError(
        "output_mode must be one of ['both', 'pdf', 'point'], got {}".format(
            output_mode
        )
    )
    spec_settings = settings["spectrum_extractor_settings"]
    test_dataloader = build_dataloader(
        settings,
        mode="test",
        cross_val_name=dataset_fold,
    )
    test_model = load_model(ckpt_path, device)
    data_info = []
    pred_pdf = []
    z_pdf = []
    pred_point = []
    z_point = []
    z_label = []
    infer_time_ms = 0.0
    total_samples = 0

    test_model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(test_dataloader),
            ncols=150,
        ) as pbar:
            for idx, batch in enumerate(test_dataloader):
                pbar.set_description("Predicting batch {}".format(idx + 1))
                (
                    object_id,
                    right_ascension,
                    declination,
                    photometric,
                    mags,
                    spe_z,
                    spe_z_origin,
                    wavelength,
                    flux,
                ) = batch
                photometric = photometric.to(device)
                mags = mags.to(device)
                spe_z = spe_z.to(device)
                spe_z_origin = spe_z_origin.to(device)
                infer_start = torch.cuda.Event(enable_timing=cuda_infer_timer)
                infer_end = torch.cuda.Event(enable_timing=cuda_infer_timer)
                for i in range(photometric.shape[0]):
                    data_info.append(
                        {
                            "object_id": object_id[i],
                            "right_ascension": right_ascension[i],
                            "declination": declination[i],
                        }
                    )

                if cuda_infer_timer:
                    infer_start.record(torch.cuda.current_stream(device))
                (
                    point_estimate,
                    pdf_estimate,
                    mag_pred,
                    deep_features,
                    spec_pred,
                    spectrum_features,
                ) = test_model(photometric, mags)
                if cuda_infer_timer:
                    infer_end.record(torch.cuda.current_stream(device))
                    torch.cuda.synchronize(device)
                    infer_time_ms += infer_start.elapsed_time(infer_end)
                if output_mode in ["both", "pdf"]:
                    pred_pdf.append(pdf_estimate.detach().cpu())
                    _pred_z_pdf = z_estimate(
                        pred=pdf_estimate,
                        mode=PDF_MODE,
                        log1p=True,
                    )
                    z_pdf.append(_pred_z_pdf.detach().cpu())
                    d_z_pdf = delta_z(
                        pred=pdf_estimate,
                        label=spe_z_origin,
                        mode=PDF_MODE,
                        log1p=True,
                    )
                    pbar.set_postfix(
                        {
                            "PDF MAE": (_pred_z_pdf - spe_z).abs().mean().item(),
                            "PDF NMAD-1.48": nmad_z(d_z_pdf, factor=1.4826).item(),
                            "PDF Sigma_0.05": sigma_n(d_z_pdf, n=0.05).item(),
                            "PDF Sigma_0.15": sigma_n(d_z_pdf, n=0.15).item(),
                            "PDF Z Bias": z_bias(d_z_pdf).item(),
                        },
                        refresh=False,
                    )
                if output_mode in ["both", "point"]:
                    _p = point_estimate.squeeze(-1)
                    pred_point.append(_p.detach().cpu())
                    _pred_z_point = z_estimate(
                        pred=_p,
                        log1p=True,
                    )
                    z_point.append(_pred_z_point.detach().cpu())
                    d_z_point = delta_z(
                        pred=_p,
                        label=spe_z,
                        log1p=True,
                    )
                    pbar.set_postfix(
                        {
                            "Point MAE": (_pred_z_point - spe_z).abs().mean().item(),
                            "Point NMAD-1.48": nmad_z(d_z_point, factor=1.4826).item(),
                            "Point Sigma_0.05": sigma_n(d_z_point, n=0.05).item(),
                            "Point Sigma_0.15": sigma_n(d_z_point, n=0.15).item(),
                            "Point Z Bias": z_bias(d_z_point).item(),
                        },
                        refresh=False,
                    )
                z_label.append(spe_z_origin.detach().cpu())
                total_samples += photometric.shape[0]
                pbar.update(1)

    print("=" * 50)
    fps = total_samples / (infer_time_ms / 1000.0)
    print(
        "[Info] FPS: {:.2f}, Total Samples: {}, Total Time: {:.2f} ms".format(
            fps,
            total_samples,
            infer_time_ms,
        )
    )
    print("=" * 50)

    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    else:
        if clear_res_dir:
            shutil.rmtree(res_save_dir)
            os.makedirs(res_save_dir)
        else:
            raise FileExistsError(res_save_dir)
    data_df_header = [
        "object_id",
        "right_ascension",
        "declination",
        "spe_z",
        "pred_z_point",
        "pred_z_pdf",
    ]
    res_json = {
        "cross_val_name": dataset_fold,
        "ckpt_path": ckpt_path,
        "total_samples": int(total_samples),
        "infer_time_ms": infer_time_ms,
        "fps": fps,
        "mae_pdf": -1,
        "nmad_1.48_pdf": -1,
        "sigma_0.05_pdf": -1,
        "simga_0.15_pdf": -1,
        "z_bias_pdf": -1,
        "outline_fraction_pdf_0.15": -1,
        "mae_point": -1,
        "nmad_1.48_point": -1,
        "sigma_0.05_point": -1,
        "simga_0.15_point": -1,
        "z_bias_point": -1,
        "outline_fraction_point_0.15": -1,
    }
    z_label = torch.cat(z_label, dim=0)
    if output_mode in ["both", "pdf"]:
        z_pdf = torch.cat(z_pdf, dim=0)
        pred_pdf = torch.cat(pred_pdf, dim=0)
        print("[Info] PDF Estimation Results:")
        mae_pdf = (z_pdf - z_label).abs().mean().item()
        d_z_epoch = delta_z(
            pred=pred_pdf,
            label=z_label,
            mode=PDF_MODE,
            log1p=True,
        )
        nmad_pdf = nmad_z(
            d_z=d_z_epoch,
            factor=1.4826,
        ).item()
        sigma_005_pdf = sigma_n(
            d_z=d_z_epoch,
            n=0.05,
        ).item()
        sigma_015_pdf = sigma_n(
            d_z=d_z_epoch,
            n=0.15,
        ).item()
        z_bias_pdf = z_bias(
            d_z=d_z_epoch,
        ).item()
        outline_frac = outline_fraction(
            d_z=d_z_epoch,
            threshold=0.1,
        ).item()
        print(
            "PDF MAE: {:.4f}, PDF NMAD-1.48: {:.4f}, PDF Sigma_0.05: {:.4f}, PDF Sigma_0.15: {:.4f}, PDF Z Bias: {:.4f}, PDF Outline Fraction-0.15: {:.4f}".format(
                mae_pdf,
                nmad_pdf,
                sigma_005_pdf,
                sigma_015_pdf,
                z_bias_pdf,
                outline_frac,
            )
        )
        res_json["mae_pdf"] = mae_pdf
        res_json["nmad_1.48_pdf"] = nmad_pdf
        res_json["sigma_0.05_pdf"] = sigma_005_pdf
        res_json["sigma_0.15_pdf"] = sigma_015_pdf
        res_json["z_bias_pdf"] = z_bias_pdf
        res_json["outline_fraction_pdf_0.15"] = outline_frac
    if output_mode in ["both", "point"]:
        z_point = torch.cat(z_point, dim=0)
        pred_point = torch.cat(pred_point, dim=0)
        print("[Info] Point Estimation Results:")
        mae_point = (z_point - z_label).abs().mean().item()
        d_z_epoch = delta_z(
            pred=pred_point,
            label=z_label,
            log1p=True,
        )
        nmad_point = nmad_z(d_z_epoch, factor=1.4826).item()
        sigma_005_point = sigma_n(d_z_epoch, n=0.05).item()
        sigma_015_point = sigma_n(d_z_epoch, n=0.15).item()
        z_bias_point = z_bias(d_z_epoch).item()
        outline_frac = outline_fraction(d_z_epoch, threshold=0.15).item()
        print(
            "Point MAE: {:.4f}, Point NMAD-1.48: {:.4f}, Point Sigma_0.05: {:.4f}, Point Sigma_0.15: {:.4f}, Point Z Bias: {:.4f}, Point Outline Fraction-0.15: {:.4f}".format(
                mae_point,
                nmad_point,
                sigma_005_point,
                sigma_015_point,
                z_bias_point,
                outline_frac,
            )
        )
        res_json["mae_point"] = mae_point
        res_json["nmad_1.48_point"] = nmad_point
        res_json["sigma_0.05_point"] = sigma_005_point
        res_json["sigma_0.15_point"] = sigma_015_point
        res_json["z_bias_point"] = z_bias_point
        res_json["outline_fraction_point_0.1"] = outline_frac
    # save json
    with open(os.path.join(res_save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res_json, f, indent=4)
    # save data info
    data_df = []
    for i in range(len(data_info)):
        data_df.append(
            {
                "object_id": str(data_info[i]["object_id"]),
                "right_ascension": str(data_info[i]["right_ascension"]),
                "declination": str(data_info[i]["declination"]),
                "spe_z": str(z_label[i].item()),
                "pred_z_point": (
                    str(z_point[i].item())
                    if output_mode in ["both", "point"]
                    else "None"
                ),
                "pred_z_pdf": (
                    str(z_pdf[i].item()) if output_mode in ["both", "pdf"] else "None"
                ),
            }
        )
    data_df = pd.DataFrame(data_df, columns=data_df_header)
    data_df.to_csv(
        os.path.join(res_save_dir, "result.csv"),
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--clear",
        "-c",
        action="store_true",
        default=False,
        help="Clear the result directory before inference",
    )
    args.add_argument(
        "--mode",
        "-m",
        default="both",
        choices=["both", "pdf", "point"],
        help="Output mode for the inference results. Choose from 'both', 'pdf', or 'point'.",
    )
    opts = args.parse_args()

    # check model path is .ckpt file
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model path {} does not exist.".format(MODEL_PATH))
    if not MODEL_PATH.endswith(".ckpt"):
        raise ValueError("Model path must be a .ckpt file, got {}".format(MODEL_PATH))
    print("[INFO] Using device: {}".format(DEVICE))
    inference(
        settings=config,
        output_mode=opts.mode,
        ckpt_path=os.path.join(MODEL_PATH),
        dataset_fold=CROSS_FOLD,
        res_save_dir=os.path.join(
            RES_SAVE_DIR, os.path.basename(MODEL_PATH).split(".ckpt")[0], CROSS_FOLD
        ),
        clear_res_dir=opts.clear,
        device=DEVICE,
        cuda_infer_timer=True,
    )
