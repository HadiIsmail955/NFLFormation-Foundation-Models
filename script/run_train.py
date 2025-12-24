from src.train.train_seg_v1_0 import train_phase

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
        "coco_file": "dataSet/splits/train.json",
        "img_size": 1024,
        "val_split": 0.1,
        "batch_size": 4,
        "num_workers": 4,
        "sam_type": "vit_h",
        "ckpt_dir": "src/model/models",
        "unfreeze_last_blocks": 0,
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "patience": 10,
        "lr_decay_factor": 0.5,
        "min_lr": 1e-7,
        "early_stopping": True,
    }

    train_phase(cfg)

if __name__ == "__main__":
    main()