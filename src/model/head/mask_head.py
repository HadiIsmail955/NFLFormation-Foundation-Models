import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductMaskHead(nn.Module):
    def __init__(self, in_channels=256, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

    def forward(self, feat_map, queries, out_hw):
        Fmap = self.proj(feat_map)  # [B, D, H', W']
        B, D, Hs, Ws = Fmap.shape
        K = queries.shape[1]

        # flatten spatial -> [B, H'*W', D]
        tokens = Fmap.flatten(2).transpose(1, 2)  # [B, N, D], N=Hs*Ws

        # dot product: [B, K, N]
        mask_logits_small = torch.einsum("bkd,bnd->bkn", queries, tokens)

        # reshape to [B, K, H', W']
        mask_logits_small = mask_logits_small.view(B, K, Hs, Ws)

        # upsample to [B, K, H, W]
        mask_logits = F.interpolate(mask_logits_small, size=out_hw, mode="bilinear", align_corners=False)
        return mask_logits