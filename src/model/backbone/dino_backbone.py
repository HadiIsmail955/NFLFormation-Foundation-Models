import os
import sys
import urllib.request
import torch
import torch.nn as nn
import timm


class DINOBackbone(nn.Module):
    def __init__(
        self,
        dino_type="vit_b",
        ckpt_path="./model/models/dinov2_vitb14_pretrain.pth",
        unfreeze_last_blocks=0,
        pretrained=True,
    ):
        super().__init__()

        dino_arch = {
            "vit_b": "vit_base_patch14",
            "vit_l": "vit_large_patch14",
            "vit_g": "vit_giant_patch14",
        }

        self.encoder = timm.create_model(
            dino_arch[dino_type],
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        if pretrained:
            state = torch.load(ckpt_path, map_location="cpu")
            self.encoder.load_state_dict(state, strict=False)

        self.embed_dim = self.encoder.num_features

        for p in self.encoder.parameters():
            p.requires_grad = False

        if unfreeze_last_blocks > 0:
            blocks = getattr(self.encoder, "blocks", None)
            assert blocks is not None, "DINO encoder has no transformer blocks"

            assert unfreeze_last_blocks <= len(blocks), (
                f"unfreeze_last_blocks={unfreeze_last_blocks} "
                f"exceeds number of blocks={len(blocks)}"
            )

            for block in blocks[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

                for m in block.modules():
                    if isinstance(m, nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True

    def forward(self, x):
        feats = self.encoder(x)
        if feats.ndim == 3:
            feats = feats[:, 0]

        return feats  # [B, C]
