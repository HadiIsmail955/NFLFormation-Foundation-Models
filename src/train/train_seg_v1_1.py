import numpy as np
import cv2
import torch

from torch.utils.data import DataLoader, random_split
from ultralytics import YOLO
from scipy import ndimage as ndi

from src.data_loader.custom_data.PresnapDataset import PresnapDataset
from src.data_loader.transformations.SAMTransformer import SegTransform
from src.data_loader.collate import presnap_collate_fn
from src.model.AutomaticMaskGenerator import AutomaticMaskGenerator
from src.utils.mask_applier import visualize_instances


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def log(msg, logger=None):
    if logger is not None:
        logger.logger.info(msg)
    else:
        print(msg)


def visualize_player_bboxes(image, boxes, scores=None):
    vis = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if scores is not None:
            label = f"{scores[i]:.2f}"
            cv2.putText(
                vis, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2
            )
    return vis


def visualize_player_centers(image, points):
    vis = image.copy()
    for (x, y) in points:
        cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 255), -1)
        cv2.circle(vis, (int(x), int(y)), 11, (0, 0, 0), 2)
    return vis


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.35):
    """
    Overlay a binary mask on an RGB image.
    """
    overlay = image.copy()

    if mask.ndim == 3:
        mask = mask.squeeze(0)

    mask = mask > 0

    overlay[mask] = (
        overlay[mask] * (1 - alpha) +
        np.array(color) * alpha
    ).astype(np.uint8)

    return overlay


# -------------------------------------------------
# CORE: SIMPLE & CORRECT PLAYER CENTERS
# -------------------------------------------------

def extract_player_centers_from_mask(
    offense_mask,
    min_distance=100,
):
    """
    Baseline, stable player center extraction using distance transform.
    """

    if offense_mask.ndim == 3:
        offense_mask = offense_mask.squeeze(0)

    mask = (offense_mask > 0).astype(np.uint8)

    if mask.sum() == 0:
        return np.empty((0, 2), dtype=np.int32), None

    # Distance transform
    distance = ndi.distance_transform_edt(mask)

    # Local maxima via dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (min_distance, min_distance)
    )
    dilated = cv2.dilate(distance, kernel)

    peaks = (distance == dilated) & (distance > 0)

    coords = np.column_stack(np.where(peaks))
    points = np.array([[c, r] for r, c in coords], dtype=np.int32)

    return points, distance

def blur_everything_except_mask(
    image_np,
    mask_np,
    blur_ksize=31,
    blur_sigma=0
):
    """
    Blur entire image except masked region.
    image_np: (H, W, 3) uint8
    mask_np: (H, W) or (1, H, W), values {0,1} or {0,255}
    """

    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(0)

    # Ensure binary uint8 mask (0 or 255)
    if mask_np.max() == 1:
        mask = (mask_np * 255).astype(np.uint8)
    else:
        mask = mask_np.astype(np.uint8)

    # Blur entire image
    blurred = cv2.GaussianBlur(
        image_np,
        (blur_ksize, blur_ksize),
        blur_sigma
    )

    # Invert mask
    inv_mask = cv2.bitwise_not(mask)

    # Foreground (original where mask==255)
    fg = cv2.bitwise_and(image_np, image_np, mask=mask)

    # Background (blurred where mask==0)
    bg = cv2.bitwise_and(blurred, blurred, mask=inv_mask)

    return cv2.add(fg, bg)

def keep_only_mask(image_np, mask_np):
    """
    Keep only masked region, set everything else to black.

    image_np: (H, W, 3) uint8
    mask_np: (H, W) or (1, H, W), values {0,1} or {0,255}
    """

    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(0)

    # Ensure binary uint8 mask (0 or 255)
    if mask_np.max() == 1:
        mask = (mask_np * 255).astype(np.uint8)
    else:
        mask = mask_np.astype(np.uint8)

    # Apply mask
    return cv2.bitwise_and(image_np, image_np, mask=mask)

# -------------------------------------------------
# Main visualization pipeline
# -------------------------------------------------

def train_phase(cfg, logger=None):
    log("Starting visualization pipeline...", logger)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}", logger)

    # -----------------------------
    # Dataset
    # -----------------------------
    seg_tf = SegTransform()

    dataset = PresnapDataset(
        data_source=cfg["data_root"],
        coco_file=cfg["train_coco_file"],
        seg_transform=seg_tf,
        classifier_transform=None,
        enable_flip=False,
    )

    val_len = int(len(dataset) * cfg["val_split"])
    train_len = len(dataset) - val_len

    _, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=presnap_collate_fn,
    )

    # -----------------------------
    # SAM
    # -----------------------------
    log("Loading SAM AutomaticMaskGenerator...", logger)

    amg = AutomaticMaskGenerator(
        sam_type=cfg["sam_type"],
        ckpt_dir=cfg["ckpt_dir"],
        unfreeze_last_blocks=0,
    )
    amg.sam.to(device)

    # -----------------------------
    # YOLO
    # -----------------------------
    log("Loading YOLOv8...", logger)

    yolo = YOLO("yolov8x.pt")
    yolo.to(device)
    yolo.eval()

    # -----------------------------
    # One sample
    # -----------------------------
    batch = next(iter(val_loader))

    image = batch["image"][0]
    offense_mask = batch["mask"][0]
    center = batch["center_map"][0]

    image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    offense_mask_np = offense_mask.squeeze(0).cpu().numpy()
    center_np = center.squeeze(0).cpu().numpy()

    cv2.imwrite("center_map.png",
        cv2.cvtColor(
            (center_np * 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    )


    # -----------------------------
    # Player centers (BASELINE)
    # -----------------------------
    player_points, distance = extract_player_centers_from_mask(
        offense_mask_np,
        min_distance=20
    )

    log(f"Detected {len(player_points)} player centers", logger)

    # -----------------------------
    # Overlay mask + centers
    # -----------------------------
    mask_overlay = overlay_mask(
        image_np,
        offense_mask_np,
        color=(0, 255, 0),
        alpha=0.35
    )

    center_vis = visualize_player_centers(
        mask_overlay,
        player_points
    )

    cv2.imwrite(
        "player_centers.png",
        cv2.cvtColor(center_vis, cv2.COLOR_RGB2BGR)
    )

    # -----------------------------
    # SAM masks
    # -----------------------------
    image_for_sam = keep_only_mask(
        image_np,
        offense_mask_np
    )

    masks = amg.generate_masks(image_for_sam)
    sam_vis = visualize_instances(image_np, masks)

    cv2.imwrite(
        "sam_masks.png",
        cv2.cvtColor(sam_vis, cv2.COLOR_RGB2BGR)
    )

    # -----------------------------
    # YOLO boxes
    # -----------------------------
    with torch.no_grad():
        results = yolo(image_np, conf=0.3, iou=0.4, verbose=False)

    det = results[0]
    if det.boxes is not None:
        boxes = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        mask = classes == 0
        person_boxes = boxes[mask]
        person_scores = scores[mask]
    else:
        person_boxes = np.empty((0, 4))
        person_scores = None

    bbox_vis = visualize_player_bboxes(
        image_np, person_boxes, person_scores
    )

    cv2.imwrite(
        "yolo_player_bboxes.png",
        cv2.cvtColor(bbox_vis, cv2.COLOR_RGB2BGR)
    )

    # -----------------------------
    # Side-by-side comparison
    # -----------------------------
    h = min(sam_vis.shape[0], bbox_vis.shape[0], center_vis.shape[0])

    def resize_keep(img):
        return cv2.resize(
            img, (int(img.shape[1] * h / img.shape[0]), h)
        )

    comparison = np.concatenate(
        [
            resize_keep(sam_vis),
            resize_keep(bbox_vis),
            resize_keep(center_vis),
        ],
        axis=1
    )

    cv2.imwrite(
        "sam_yolo_centers.png",
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    )


    log("Saved outputs:", logger)
    log("  sam_masks.png", logger)
    log("  yolo_player_bboxes.png", logger)
    log("  player_centers.png", logger)
    log("  sam_yolo_centers.png", logger)
