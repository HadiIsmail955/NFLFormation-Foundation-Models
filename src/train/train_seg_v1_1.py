import math

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import autocast
from torch.amp import GradScaler

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.AutomaticMaskGenerator import AutomaticMaskGenerator

from src.utils.losses import make_loss
from src.utils.metrics import dice_iou_from_logits
from src.data_loader.collate import presnap_collate_fn
from src.utils.mask_applier import visualize_instances
import matplotlib.pyplot as plt
# from src.utils.seed import set_seed


def log(msg, logger):
    # print(msg)
    logger.logger.info(msg)


def train_phase(cfg, logger):
    # set_seed(cfg["seed"])
    log("Initializing training...", logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    log(f"Using device: {device}", logger)

    log("Loading dataset metadata...", logger)

    seg_tf = SegTransform()

    dataset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["train_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=cfg["flip_augmentation"],
        flip_prob=cfg["flip_prob"],
        )

    log(f"Dataset loaded: {len(dataset)} samples", logger)

    log("Splitting dataset into train / validation...", logger)

    val_len = int(len(dataset) * cfg["val_split"])
    train_len = len(dataset) - val_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    log(f"Split done -> train={train_len}, val={val_len}", logger)

    log("Creating DataLoaders...", logger)

    persistent = cfg["num_workers"] > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=persistent,
        collate_fn=presnap_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=persistent,
        collate_fn=presnap_collate_fn,
    )

    log(
        f"DataLoaders ready "
        f"(batch_size={cfg['batch_size']}, workers={cfg['num_workers']})",
        logger,
    )

    log("Initializing model...", logger)

    model = AutomaticMaskGenerator(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
    ).to(device)

    if cfg.get("continue_from_ckpt", None) is not None:
        state = torch.load(cfg.get("continue_from_ckpt", None), map_location=device)
        model.load_state_dict(state["model"])
        log(
            f"Continuing training from checkpoint: "
            f"{cfg.get('continue_from_ckpt', None) is not None}",
            logger,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model ready ({trainable_params:,} trainable parameters)", logger)

    # assert trainable_params < 6_000_000, "Encoder accidentally unfrozen!"

    batch = next(iter(val_loader))

    # seg_image is a torch tensor [B, 3, H, W]
    seg_image = batch["seg_image"][0]

    image_np = (
        seg_image
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    image_np = (image_np * 255).astype(np.uint8)

    masks = model.generate_masks(image_np)

    log(f"Generated {len(masks)} masks for sample image", logger)

    vis = visualize_instances(image_np, masks)
    vis_path = cfg.get("mask_vis_path", "mask_vis.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    log(f"Saved mask visualization to: {vis_path}", logger)

    log(f"Training finished. Best Dice: {best_dice:.4f}", logger)
    logger.close()
