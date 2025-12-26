import numpy as np
import cv2

def save_overlay_with_metrics(
    image,
    gt_mask,
    pred_mask,
    metrics,
    save_path,
):

    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    gt = gt_mask.squeeze().cpu().numpy()
    pred = pred_mask.squeeze().cpu().numpy()

    overlay = img.copy()
    overlay[gt > 0.5] = overlay[gt > 0.5] * 0.7 + np.array([0, 255, 0]) * 0.3
    overlay[pred > 0.5] = overlay[pred > 0.5] * 0.7 + np.array([255, 0, 0]) * 0.3

    overlay = overlay.astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    color = (255, 255, 255)

    y0 = 22
    dy = 20

    lines = [
        f"Dice: {metrics['dice']:.3f}",
        f"IoU:  {metrics['iou']:.3f}",
        f"Prec: {metrics['precision']:.3f}",
        f"Rec:  {metrics['recall']:.3f}",
    ]

    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(
            overlay,
            line,
            (10, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, overlay[..., ::-1])
