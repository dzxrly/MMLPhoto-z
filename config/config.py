config = {
    # dev mode
    "debug": False,  # pytorch lightning trainer fast_dev_run
    "wandb_project_name": "DESI_COSMOS2020_PARAMS",
    "enable_wandb": True,  # enable wandb
    "wandb_params_search": True,
    "enable_random_seed_search": False,
    "verbose": True,
    "run_test_when_training_end": True,
    # training settings
    "random_seed": 42,
    "used_device": [0, 1, 2, 3],
    "precision": "32-true",
    "dataset_root_dir": "/home/eggtargaryen/dataset/Euclid/DESI_WISE/dataset",
    # dataset settings
    "photometric_dir_name": "photo",
    "label_dir_name": "label",
    "augmentation": False,
    "mask_channels": 1,
    "mask_prob_in_channel": 0.25,
    "masked_replace": 0.0,
    # torch 2.0
    "enable_torch_2": False,
    # parameters
    "photo_in_channel": 6,  # DESI g r i z + WISE W1 W2
    "photo_in_size": 64,
    "photo_min_max": (-1.4950683, 710.1159),
    "extractor_out_dim": 512,
    "mag_in_size": 5 + 4,  # mag + mag diff
    "out_gaussian_groups": 5,
    "z_categories": 40,
    # others
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "monitor": "val_mae_epoch",
    "min_delta": 0.002,
    "mode": "min",
    "eps": 1e-12,
    "patience": 150,
    "gradient_clip_val": 20,
    # model settings
    "batch_size": 32,
    "num_workers": 32,
    "epochs": 1000,
    "learn_rate": 0.0001,
    "cos_annealing_t_0": 10,
    "cos_annealing_t_mult": 2,
    "cos_annealing_eta_min": 1e-12,
}
