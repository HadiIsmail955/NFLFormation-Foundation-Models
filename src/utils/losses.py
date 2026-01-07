import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def make_loss():
    return BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

def soft_masks_iou(pred_probs, gt_masks, eps=1e-6):
    K, H, W = pred_probs.shape
    G = gt_masks.shape[0]

    pred_f = pred_probs.view(K, -1)
    gt_f = gt_masks.float().view(G, -1)

    inter = pred_f @ gt_f.t()
    union = pred_f.sum(1, keepdim=True) + gt_f.sum(1).unsqueeze(0) - inter
    return inter / (union + eps)

def dice_bce_loss(pred_logits, gt_masks, eps=1e-6):

    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks.float())

    pred = pred_logits.sigmoid()
    num = 2 * (pred * gt_masks).sum(dim=(1,2))
    den = pred.sum(dim=(1,2)) + gt_masks.sum(dim=(1,2)) + eps
    dice = 1 - (num / den).mean()

    return bce + dice

def hungarian_match(cost_matrix):
    cost = cost_matrix.detach().cpu()
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.numpy())
        return torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)
    except Exception:
        K, G = cost.shape
        used_g = set()
        pred_ids, gt_ids = [], []
        for k in range(K):
            g = int(torch.argmin(cost[k]).item())
            if g not in used_g:
                used_g.add(g)
                pred_ids.append(k)
                gt_ids.append(g)
                if len(used_g) == G:
                    break
        return torch.tensor(pred_ids, dtype=torch.long), torch.tensor(gt_ids, dtype=torch.long)

def compute_losses(
    outputs,
    playerMasks,
    roles,
    formation_labels,
    lambda_role=0.5,
    lambda_pres=0.2,
    lambda_form=0.3,
):
    mask_logits = outputs["mask_logits"]
    role_logits = outputs["role_logits"]
    pres_logits = outputs["present_logits"]
    formation_logits = outputs["formation_logits"]

    B, K, H, W = mask_logits.shape

    total_mask_loss = 0.0
    total_role_loss = 0.0
    total_pres_loss = 0.0
    total_form_loss = 0.0

    for b in range(B):
        gt_masks = torch.stack(playerMasks[b]).squeeze(1).to(mask_logits.device)  # [G,H,W]
        gt_roles = torch.as_tensor(roles[b], device=role_logits.device)       # [G]
        G = gt_masks.shape[0]

        pres_tgt = torch.zeros((K,), device=mask_logits.device)

        if G > 0:
            pred_probs = mask_logits[b].sigmoid()
            iou = soft_masks_iou(pred_probs, gt_masks)
            cost = 1.0 - iou

            pred_idx, gt_idx = hungarian_match(cost)
            pred_idx = pred_idx.to(mask_logits.device)
            gt_idx = gt_idx.to(mask_logits.device)

            total_mask_loss += dice_bce_loss(
                mask_logits[b, pred_idx],
                gt_masks[gt_idx]
            )

            total_role_loss += F.cross_entropy(
                role_logits[b, pred_idx],
                gt_roles[gt_idx]
            )

            pres_tgt[pred_idx] = 1.0

        total_pres_loss += F.binary_cross_entropy_with_logits(
            pres_logits[b, :, 0], pres_tgt
        )
        counts = Counter({
            'trips-right': 72, 'shotgun': 71, 'trips-left': 63, 'empty': 54,
            'ace-left': 45, 'trey-left': 44, 'bunch-right': 42, 'trey-right': 42,
            'bunch-left': 39, 'heavy': 39, 'double-tight': 36, 'ace-right': 31,
            'i-formation': 27, 'twins-right': 22
        })

        labels = list(counts.keys())
        freqs = torch.tensor([counts[l] for l in labels], dtype=torch.float)

        weights = 1.0 / torch.log1p(freqs)
        weights = weights / weights.mean()

        formation_loss_fn = nn.CrossEntropyLoss(weight=weights.to(formation_logits.device))

        total_form_loss += formation_loss_fn(
            formation_logits[b].unsqueeze(0),
            formation_labels[b].unsqueeze(0)
        )


    total_mask_loss /= B
    total_role_loss /= B
    total_pres_loss /= B
    total_form_loss /= B

    total = (
        total_mask_loss
        + lambda_role * total_role_loss
        + lambda_pres * total_pres_loss
        + lambda_form * total_form_loss
    )

    return total, {
        "mask_loss": total_mask_loss.detach(),
        "role_loss": total_role_loss.detach(),
        "presence_loss": total_pres_loss.detach(),
        "formation_loss": total_form_loss.detach(),
    }
