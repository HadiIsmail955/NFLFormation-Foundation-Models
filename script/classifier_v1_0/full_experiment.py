from src.train.train_classifier_v1_0 import train_phase
from src.test.test_classifier_v1_0 import test_phase
from src.utils.experiment_logger import ExperimentLogger


def main():
    cfg = {
        "experiment_name": "nfl_formation_classifier_v1_0",
        "experiment_description": (
            "Training classifier model for NFL formation detection "
            "using SAM + DINOv2. Version 1.0."
        ),
        "experiment_goals": [
            "Achieve high accuracy in classifying NFL formations",
            "Utilize SAM backbone for spatial grounding",
            "Leverage DINOv2 for strong visual representations",
        ],
        "seed": 42,
        "data_root": "dataSet",
        "train_coco_file": "dataSet/splits/train.json",
        "test_coco_file": "dataSet/splits/test.json",
        "seg_from_ckpt": "outputs/seg_phase_v1-0_20251230_150843/best.pt",
        "img_size": 1024,
        "val_split": 0.1,
        "batch_size": 4,
        "num_workers": 4,
        "sam_type": "vit_h",
        "sam_ckpt_dir": "src/model/models",
        "unfreeze_last_blocks": 2,
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "patience": 10,
        "lr_decay_factor": 0.3,
        "min_lr": 1e-7,
        "threshold": 1e-3,
        "dino_type": "vit_b",
        "num_classes": 14,
        "mask_mode": "soft",
        "flip_augmentation": False,
        "flip_prob": 0.5,
    }
    train=True
    logger = ExperimentLogger(exp_name="seg_phase_v1-0")
    logger.save_config(cfg)
    if train:
        train_phase(cfg, logger)
    test_phase(cfg, logger)


if __name__ == "__main__":
    main()