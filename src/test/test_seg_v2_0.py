import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import autocast

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMSegmenter_v1_0 import SAMSegmenter
from src.data_loader.collate import presnap_collate_fn
from src.utils.merge_image import save_overlay_with_metrics
from src.utils.mask_applier import blur_outside_sharpen_inside_mask



@torch.no_grad()
def test_phase(cfg, logger):
    logger.logger.info("Initializing testing (center heatmap task)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    logger.logger.info(f"Using device: {device}")

    # -----------------------
    # Dataset
    # -----------------------
    seg_tf = SegTransform()

    dataset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["test_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=cfg["flip_augmentation"],
        flip_prob=cfg["flip_prob"],
    )

    logger.logger.info(f"Test dataset loaded: {len(dataset)} samples")

    test_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device == "cuda"),
        persistent_workers=cfg["num_workers"] > 0,
        collate_fn=presnap_collate_fn,
    )

    # -----------------------
    # Model
    # -----------------------
    model = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=0,
    ).to(device)

    ckpt_path = logger.get_best_checkpoint_path()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    logger.logger.info(f"Loaded checkpoint: {ckpt_path}")

    # -----------------------
    # Loss
    # -----------------------
    loss_fn = torch.nn.BCEWithLogitsLoss()


    # -----------------------
    # Metrics accumulators
    # -----------------------
    total_loss = 0.0
    total_peak_err = 0.0
    sample_idx = 0


    # -----------------------
    # Testing loop
    # -----------------------
    test_bar = tqdm(test_loader, desc="Testing", leave=False)
    vis_dir = logger.get_viz_dir()
    os.makedirs(vis_dir, exist_ok=True)

    for batch in test_bar:
        x = batch["seg_image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["center_map"].to(device, non_blocking=True)
        # x=blur_outside_sharpen_inside_mask(x,mask)

        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(x)
            loss = loss_fn(logits, y)

        total_loss += loss.item()

        # -------- Peak localization error --------
        # distance between predicted & GT max (pixels)
        B = x.size(0)
        for b in range(B):
            mse = loss_fn(logits[b], y[b]).item()

            py, px = torch.unravel_index(logits[b,0].argmax(), logits[b,0].shape)
            ty, tx = torch.unravel_index(y[b,0].argmax(), y[b,0].shape)

            peak_err = torch.sqrt(
                (px - tx).float() ** 2 + (py - ty).float() ** 2
            ).item()
            dist = torch.sqrt(
                (px - tx).float() ** 2 + (py - ty).float() ** 2
            )
            total_peak_err += dist.item()
            
            metrics = {
                "mse": mse,
                "peak_error_px": peak_err,
            }
            
            save_overlay_with_metrics(
                x[b],
                y[b],
                logits[b],
                metrics,
                os.path.join(vis_dir, f"sample_{sample_idx:04d}.png"),
            )
            sample_idx += 1

    # -----------------------
    # Final metrics
    # -----------------------
    n_batches = len(test_loader)

    metrics = {
        "test_mse": total_loss / n_batches,
        "test_peak_error_px": total_peak_err / max(sample_idx, 1),
    }

    for k, v in metrics.items():
        logger.logger.info(f"{k}: {v:.6f}")

    logger.logger.info(
        "Test results | "
        f"MSE={metrics['test_mse']:.6f}, "
        f"peak_error(px)={metrics['test_peak_error_px']:.2f}"
    )

    logger.close()
