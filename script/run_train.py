from src.train.train_seg_v1_0 import train_phase

def main():
    cfg = {
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
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "patience": 10,
    }

    train_phase(cfg)

if __name__ == "__main__":
    main()