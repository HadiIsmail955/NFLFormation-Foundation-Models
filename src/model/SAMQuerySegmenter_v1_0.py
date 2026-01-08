import os
import sys
import torch
import urllib.request
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

from .backbone.sam_backbone import SAMBackbone
from .decoder.query_instance_decoder import QueryInstanceDecoder
from .head.mask_head import DotProductMaskHead
from .head.role_head import RoleHead
from .head.presence_head import PresenceHead
from .head.basic_formation_head import FormationHead

from src.utils.mask_applier import compute_centers_from_masks

class SAMQuerySegmenter(nn.Module):
    def __init__(
        self,
        sam_type="vit_h", 
        ckpt_dir="./model/models", 
        unfreeze_last_blocks=0,
        d_model=256,
        num_queries=11,
        num_roles=8,
        dec_layers=4,
        dec_heads=8
    ):
        super().__init__()
        sam_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        
        sam_checkpoint = f"sam_{sam_type}.pth"
        
        ckpt_root = ckpt_dir
        os.makedirs(ckpt_root, exist_ok=True)

        ckpt_path = os.path.join(ckpt_root, sam_checkpoint)

        if not os.path.exists(ckpt_path):
            url = sam_urls[sam_type]
            print(f"Downloading SAM checkpoint ({sam_type}) from {url} ...")

            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                progress = min(int(downloaded / total_size * 100), 100)
                sys.stdout.write(f"\rDownloading: {progress}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, ckpt_path, reporthook=show_progress)
            print("\nDownload complete!")

        sam = sam_model_registry[sam_type](checkpoint=ckpt_path)


        self.backbone = SAMBackbone(
            sam=sam,
            unfreeze_last_blocks=unfreeze_last_blocks
        )
        self.num_queries = num_queries

        self.query_decoder = QueryInstanceDecoder(
            d_model=d_model, nhead=dec_heads, num_layers=dec_layers, num_queries=num_queries
        )
        self.mask_head = DotProductMaskHead(in_channels=d_model, d_model=d_model)
        self.role_head = RoleHead(d_model=d_model, num_roles=num_roles)
        self.presence_head = PresenceHead(d_model=d_model)
        self.formation_head = FormationHead(d_model=d_model+2, num_formations=14)

    def forward(self, x, offense_mask=None):
        B, _, H, W = x.shape
        feat = self.backbone(x)  # expect [B, C=256, H', W']

        if offense_mask is not None:
            m = offense_mask.float()  # [B,1,H,W]
            m = F.interpolate(m, size=feat.shape[-2:], mode="nearest")     # [B,1,H',W']
            feat = feat * m

        tokens = feat.flatten(2).transpose(1, 2)  # [B, N, D]

        Q = self.query_decoder(tokens)            # [B, K, D]

        mask_logits = self.mask_head(feat, Q, out_hw=(H, W))     # [B, K, H, W]
        role_logits = self.role_head(Q)                          # [B, K, R]
        present_logits = self.presence_head(Q)                   # [B, K, 1]
        
        centers = compute_centers_from_masks(mask_logits, H, W)
        centers = centers.to(Q.device).float()
        Q_aug = torch.cat([Q, centers], dim=-1)
        formation_logits = self.formation_head(Q_aug, present_logits) # [B, num_formations]

        return {
            "mask_logits": mask_logits,
            "role_logits": role_logits,
            "present_logits": present_logits,
            "formation_logits": formation_logits
        }
    