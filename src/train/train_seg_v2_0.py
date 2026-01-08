import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import autocast
from torch.amp import GradScaler

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMSegmenter_v1_0 import SAMSegmenter
from src.data_loader.collate import presnap_collate_fn
from src.utils.mask_applier import blur_outside_sharpen_inside_mask

from src.utils.losses import make_loss



def log(msg, logger):
    logger.logger.info(msg)


def train_phase(cfg, logger):
    log("Initializing training...", logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda")
    log(f"Using device: {device}", logger)

    # -----------------------
    # Dataset
    # -----------------------
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

    val_len = int(len(dataset) * cfg["val_split"])
    train_len = len(dataset) - val_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    log(f"Split done -> train={train_len}, val={val_len}", logger)

    # -----------------------
    # DataLoaders
    # -----------------------
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

    # -----------------------
    # Model
    # -----------------------
    model = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
    ).to(device)

    if cfg.get("continue_from_ckpt") is not None:
        state = torch.load(cfg["continue_from_ckpt"], map_location=device)
        model.load_state_dict(state["model"])
        log("Continuing training from checkpoint", logger)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model ready ({trainable_params:,} trainable parameters)", logger)

    # -----------------------
    # Optim / Scheduler
    # -----------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",                      # IMPORTANT
        factor=cfg["lr_decay_factor"],
        patience=math.ceil(cfg["patience"] // 2),
        threshold=cfg["threshold"],
        threshold_mode="rel",
        min_lr=cfg["min_lr"],
        verbose=True,
    )

    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = make_loss()

    scaler = GradScaler(
        device="cuda",
        enabled=amp_enabled,
    )

    # -----------------------
    # Training Loop
    # -----------------------
    best_val_loss = float("inf")
    patience_counter = 0

    log("Starting training loop...", logger)

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"\nEpoch {epoch}/{cfg['epochs']} | lr={current_lr:.6f}", logger)

        # -------- Train --------
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Train]",
            leave=False,
        )

        for batch in train_bar:
            x = batch["seg_image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["center_map"].to(device, non_blocking=True)
            # x=blur_outside_sharpen_inside_mask(x,mask)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(x)
                loss = loss_fn(logits, y)

            if not torch.isfinite(loss):
                log("Non-finite loss detected. Stopping.", logger)
                logger.close()
                return

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                cfg["grad_clip"],
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss /= len(train_loader)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for batch in val_bar:
                x = batch["seg_image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                y = batch["center_map"].to(device, non_blocking=True)
                x=blur_outside_sharpen_inside_mask(x,mask)

                with autocast(device_type="cuda", enabled=amp_enabled):
                    logits = model(x)
                    loss = loss_fn(logits, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # -------- Logging --------
        logger.log_epoch({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_mse": val_loss,
        })

        log(
            f"Epoch {epoch} summary | "
            f"train_mse={train_loss:.6f}, "
            f"val_mse={val_loss:.6f}",
            logger,
        )

        # -------- Scheduler / Early Stop --------
        previous_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.save_checkpoint(
                model,
                name="best.pt",
                epoch=epoch,
                val_loss=best_val_loss,
            )
            log("New best model saved", logger)
        elif current_lr < previous_lr:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] and cfg.get("early_stopping", True):
            log("Early stopping triggered", logger)
            break

    log(f"Training finished. Best val MSE: {best_val_loss:.6f}", logger)
    logger.close()
