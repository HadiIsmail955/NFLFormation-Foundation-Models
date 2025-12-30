import os
import sys
import urllib.request
import torch.nn as nn
from .backbone.dino_backbone import DINOBackbone
from .head.classification_head import ClassificationHead

class DINOClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        dino_type="vit_b",
        ckpt_dir="./model/models",
        pretrained=True,
        unfreeze_last_blocks=0,
        dropout=0.1,
    ):
        super().__init__()

        dino_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14_pretrain.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14_pretrain.pth",
            "vit_g": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14_pretrain.pth",
        }

        assert dino_type in dino_urls, f"Unknown dino_type: {dino_type}"

        ckpt_root = ckpt_dir
        os.makedirs(ckpt_root, exist_ok=True)

        ckpt_name = f"dinov2_{dino_type}.pth"
        ckpt_path = os.path.join(ckpt_root, ckpt_name)

        if pretrained and not os.path.exists(ckpt_path):
            url = dino_urls[dino_type]
            print(f"Downloading DINOv2 checkpoint ({dino_type}) from {url} ...")

            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                progress = min(int(downloaded / total_size * 100), 100)
                sys.stdout.write(f"\rDownloading: {progress}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(url, ckpt_path, reporthook=show_progress)
            print("\nDownload complete!")

        self.backbone = DINOBackbone(
            dino_type=dino_type,
            ckpt_path=ckpt_path,
            pretrained=pretrained,
            unfreeze_last_blocks=unfreeze_last_blocks,
        )
        self.classifier = ClassificationHead(
            in_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        feats = self.backbone(x)   # [B, C]
        logits = self.classifier(feats)  # [B, num_classes]
        return logits
