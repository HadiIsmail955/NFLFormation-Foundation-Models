import torch

def apply_mask(image, mask, mode="soft"):

    if mode == "hard":
        return image * (mask > 0.5)

    elif mode == "soft":
        return image * (0.7 + 0.3 * mask)


    elif mode == "background_gray":
        gray = image.mean(dim=1, keepdim=True)
        return image * mask + gray * (1 - mask)

    else:
        raise ValueError(f"Unknown mask mode: {mode}")
    
def compute_centers_from_masks(mask_logits, H, W, eps=1e-6):
    B, K, _, _ = mask_logits.shape
    probs = mask_logits.sigmoid()

    ys = torch.linspace(0, 1, H, device=mask_logits.device).view(1, 1, H, 1)
    xs = torch.linspace(0, 1, W, device=mask_logits.device).view(1, 1, 1, W)

    mass = probs.sum(dim=(2,3), keepdim=True) + eps
    x = (probs * xs).sum(dim=(2,3), keepdim=True) / mass
    y = (probs * ys).sum(dim=(2,3), keepdim=True) / mass

    centers = torch.cat([x, y], dim=-1).squeeze(2).squeeze(2)  # [B,K,2]
    return centers
