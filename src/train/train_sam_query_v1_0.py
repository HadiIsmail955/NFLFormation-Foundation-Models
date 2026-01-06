import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
from torch.amp import GradScaler

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.model.SAMSegmenter_v1_0 import SAMSegmenter
from src.model.SAMQuerySegmenter_v1_0 import SAMQuerySegmenter

from src.utils.metrics import dice_iou_from_logits
from src.utils.losses import compute_losses, hungarian_match, soft_masks_iou
from src.data_loader.collate import presnap_collate_fn
# from src.utils.seed import set_seed

from sklearn.metrics import f1_score


def log(msg, logger):
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

    val_len = int(len(dataset) * cfg["val_split"])
    train_len = len(dataset) - val_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    log(f"Split done -> train={train_len}, val={val_len}", logger)

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

    log("Initializing SAM model...", logger)

    sam_model = SAMSegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["sam_ckpt_dir"],
        unfreeze_last_blocks=0,
    ).to(device)

    assert isinstance(cfg.get("seg_from_ckpt", None), str), \
        "seg_from_ckpt must be a valid checkpoint path"

    state = torch.load(cfg["seg_from_ckpt"], map_location=device)
    sam_model.load_state_dict(state.get("model", state), strict=True)

    log(f"Loaded SAM model from: {cfg['seg_from_ckpt']}", logger)

    sam_model.eval()
    for p in sam_model.parameters():
        p.requires_grad = False
    
    model = SAMQuerySegmenter(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=cfg["unfreeze_last_blocks"],
        d_model=cfg.get("d_model", 256),
        num_queries=cfg.get("num_queries", 11),
        num_roles=cfg.get("num_roles", 8),
        dec_layers=cfg.get("dec_layers", 4),
        dec_heads=cfg.get("dec_heads", 8),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          
        factor=cfg["lr_decay_factor"],          
        patience=math.ceil(cfg["patience"] // 2),          
        threshold=cfg["threshold"],
        threshold_mode="rel",
        min_lr=cfg["min_lr"],
        verbose=True,
    )

    scaler = GradScaler(
        device="cuda",
        enabled=amp_enabled,
    )

    log("Starting training loop...", logger)

    lambda_form = 0.0
    best_score = -1e9
    patience_counter = 0
    
    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        log(f"\nEpoch {epoch}/{cfg['epochs']} | lr={current_lr:.6f}", logger)

        model.train()
        sam_model.eval()

        if epoch == cfg["formation_start_epoch"]:
            lambda_form = cfg["lambda_form"]
            log("Formation loss ENABLED", logger)
        
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [Train]", leave=False)

        for batch in train_bar:
            x = batch["seg_image"].to(device, non_blocking=True)
            playerMasks = batch["playerMasks"]
            roles = batch["roles"]
            formation_labels = batch["formation_label"].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=amp_enabled):
                offense_mask = sam_model(x)
                outputs = model(x, offense_mask)
                loss, loss_dict = compute_losses(
                    outputs,
                    playerMasks,
                    roles,
                    formation_labels,
                    lambda_form=lambda_form,
                    lambda_role=cfg.get("lambda_role", 0.5),
                    lambda_pres=cfg.get("lambda_pres", 0.2),
                )

            if not torch.isfinite(loss):
                log("Non-finite loss detected. Stopping training.", logger)
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
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lf=f"{lambda_form:.2f}")

        train_loss /= len(train_loader)

        log("Running validation...", logger)

        model.eval()
        sam_model.eval()

        val_loss = 0.0

        dice_sum, iou_sum, dice_n = 0.0, 0.0, 0

        role_y_true, role_y_pred = [], []
        form_y_true, form_y_pred = [], []

        pres_tp, pres_fp, pres_fn = 0.0, 0.0, 0.0 

        for batch in tqdm(val_loader, desc="[Val]", leave=False):
            x = batch["seg_image"].to(device, non_blocking=True)          # [B,3,H,W]
            playerMasks = batch["playerMasks"]                            # list length B
            roles = batch["roles"]                                        # list length B
            formation_labels = batch["formation_label"].to(device, non_blocking=True)  # [B]

            with autocast(device_type="cuda", enabled=amp_enabled):
                offense_mask = sam_model(x)                                   # [B,1,H,W]
                outputs = model(x, offense_mask)
                loss, _ = compute_losses(
                    outputs,
                    playerMasks,
                    roles,
                    formation_labels,
                    lambda_form=lambda_form,
                )

            val_loss += loss.item()

            form_logits = outputs["formation_logits"]  # [B,14]
            form_pred = torch.argmax(form_logits, dim=1)
            form_y_true.extend(formation_labels.detach().cpu().tolist())
            form_y_pred.extend(form_pred.detach().cpu().tolist())

            mask_logits = outputs["mask_logits"]        # [B,K,H,W]
            role_logits = outputs["role_logits"]        # [B,K,R]
            pres_logits = outputs["present_logits"]     # [B,K,1]
            B, K, H, W = mask_logits.shape

            for b in range(B):
                gt_masks = playerMasks[b].to(device)   # [G,H,W]
                gt_roles = roles[b].to(device)         # [G]
                G = gt_masks.shape[0]

                pres_tgt = torch.zeros((K,), device=device)

                if G > 0:
                    pred_probs = mask_logits[b].sigmoid()                 # [K,H,W]
                    iou_mat = soft_masks_iou(pred_probs, gt_masks)        # [K,G]
                    cost = 1.0 - iou_mat

                    pred_idx, gt_idx = hungarian_match(cost)
                    pred_idx = pred_idx.to(device)
                    gt_idx = gt_idx.to(device)

                    d, i = dice_iou_from_logits(
                        mask_logits[b, pred_idx],
                        gt_masks[gt_idx],
                        threshold=0.5,
                        reduce=True,
                    )
                    dice_sum += d
                    iou_sum += i
                    dice_n += 1

                    r_pred = torch.argmax(role_logits[b, pred_idx], dim=1)
                    role_y_true.extend(gt_roles[gt_idx].detach().cpu().tolist())
                    role_y_pred.extend(r_pred.detach().cpu().tolist())

                    pres_tgt[pred_idx] = 1.0

                p_pred = (pres_logits[b, :, 0].sigmoid() > 0.5).float()
                p_true = pres_tgt.float()

                pres_tp += float((p_pred * p_true).sum().item())
                pres_fp += float((p_pred * (1 - p_true)).sum().item())
                pres_fn += float(((1 - p_pred) * p_true).sum().item())

        val_loss /= max(len(val_loader), 1)

        mask_dice = dice_sum / max(dice_n, 1)
        mask_iou = iou_sum / max(dice_n, 1)

        role_acc = (torch.tensor(role_y_true) == torch.tensor(role_y_pred)).float().mean().item() if len(role_y_true) else 0.0
        role_macro_f1 = f1_score(role_y_true, role_y_pred, average="macro", zero_division=0) if len(role_y_true) else 0.0

        formation_acc = (torch.tensor(form_y_true) == torch.tensor(form_y_pred)).float().mean().item() if len(form_y_true) else 0.0
        formation_macro_f1 = f1_score(form_y_true, form_y_pred, average="macro", zero_division=0) if len(form_y_true) else 0.0

        pres_precision = pres_tp / max(pres_tp + pres_fp, 1e-6)
        pres_recall = pres_tp / max(pres_tp + pres_fn, 1e-6)
        pres_f1 = 2 * pres_precision * pres_recall / max(pres_precision + pres_recall, 1e-6)

        metrics = {
            "mask_dice": mask_dice,
            "mask_iou": mask_iou,
            "role_acc": role_acc,
            "role_macro_f1": role_macro_f1,
            "presence_f1": pres_f1,
            "formation_acc": formation_acc,
            "formation_macro_f1": formation_macro_f1,
        }

        logger.log_epoch({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
            "lambda_form": lambda_form,
        })

        log(
            f"Epoch {epoch} summary | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"mask_dice={metrics['mask_dice']:.4f}, mask_iou={metrics['mask_iou']:.4f}, "
            f"role_f1={metrics['role_macro_f1']:.4f}, presence_f1={metrics['presence_f1']:.4f}, "
            f"form_f1={metrics['formation_macro_f1']:.4f}",
            logger,
        )

        score = metrics["formation_macro_f1"] 
        previous_lr = current_lr
        scheduler.step(score)
        current_lr = optimizer.param_groups[0]["lr"]

        if score > best_score:
            best_score = score
            patience_counter = 0
            logger.save_checkpoint(
                model,
                name="best.pt",
                epoch=epoch,
                score=best_score,
            )
            log("New best model saved", logger)
        elif current_lr < previous_lr:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] and cfg.get("early_stopping", True):
            log("Early stopping triggered", logger)
            break

    log(f"Training finished. Best score: {best_score:.4f}", logger)
    logger.close()