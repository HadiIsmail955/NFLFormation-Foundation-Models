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
