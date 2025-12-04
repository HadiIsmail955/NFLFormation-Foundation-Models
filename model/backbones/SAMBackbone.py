import os
import sys
import torch
import urllib.request
import torch.nn as nn
from segment_anything import sam_model_registry

class SAMBackbone(nn.Module):
    """
    Extracts multi-scale feature maps from SAM's image encoder (frozen).
    Returns a list of features [C2, C3, C4, C5].
    """
    def __init__(self, sam_type="vit_h", checkpoint="./model/models"):
        super().__init__()
        sam_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        sam_checkpoint = f"sam_{sam_type}.pth"

        checkpoint= os.path.join(checkpoint,sam_checkpoint)

        # === DOWNLOAD SAM IF NEEDED ===
        if not os.path.exists(checkpoint):
            url = sam_urls[sam_type]
            print(f"⚡ Downloading SAM checkpoint ({sam_type}) from {url} ...")
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                progress = min(int(downloaded / total_size * 100), 100)
                sys.stdout.write(f"\rDownloading: {progress}%")
                sys.stdout.flush()
            urllib.request.urlretrieve(url, checkpoint, reporthook=show_progress)
            print("\n✅ Download complete!")

        sam = sam_model_registry[sam_type](checkpoint=checkpoint)
        
        self.encoder = sam.image_encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        x: preprocessed tensor (B,3,H,W)
        returns: list of feature maps, e.g. [C2,C3,C4,C5]
        """
        features = self.encoder(x)  # SAM returns multiple features internally
        return features
