from src.train.train_seg_v1_0 import train_phase
from src.test.test_seg_v1_0 import test_phase
from src.utils.experiment_logger import ExperimentLogger

def main():
    cfg = {
        "experiment_name": "nfl_formation_segmentation_v1_0",
        "experiment_description": "Training segmentation model for NFL formation detection using SAM backbone. Version 1.0 with adjusted hyperparameters.",
        "experiment_goals": [
            "Achieve high accuracy in segmenting NFL formations from images.",
            "Utilize SAM backbone for improved feature extraction.",
            "Optimize training process with effective hyperparameter tuning."
        ],
        "seed": 42,
        "data_root": "dataSet",
        "train_coco_file": "dataSet/splits/train.json",
        "test_coco_file": "dataSet/splits/test.json",
        "continue_from_ckpt": None,
        "img_size": 1024,
        "val_split": 0.1,
        "batch_size": 12,
        "num_workers": 4,
        "sam_type": "vit_h",
        "ckpt_dir": "src/model/models",
        "unfreeze_last_blocks": 0,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 5e-5,
        "grad_clip": 1.0,
        "patience": 8,
        "lr_decay_factor": 0.3,
        "min_lr": 1e-7,
        "threshold": 1e-3,
        "early_stopping": True,
        "flip_augmentation": False,
        "flip_prob": 0.5,
        "n_classes": 14,
        "k_decoder_layers": 3,
        "trainable_decoder": False,
    }
    train=True
    logger = ExperimentLogger(exp_name="seg_phase_v1-0")
    logger.save_config(cfg)
    if train:
        train_phase(cfg, logger)
    test_phase(cfg, logger)


if __name__ == "__main__":
    main()