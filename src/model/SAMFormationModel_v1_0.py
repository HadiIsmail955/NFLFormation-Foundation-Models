
import os
import sys
import urllib.request
import torch.nn as nn
from segment_anything import sam_model_registry

from .backbone.sam_backbone import SAMBackbone
from .prompt.sam_prompt_encoder import SAMPromptEncoder
from src.model.decoder.sam_classification_decoder import SAMClassificationDecoder
from src.model.utils.geometry_enhancer import GeometryEnhancer
from src.model.head.formation_head import FormationHead

class SAMFormationModel(nn.Module):
    def __init__(self, sam_type="vit_h", ckpt_dir="./model/models", unfreeze_last_blocks=0, n_classes=20, k_decoder_layers=3, trainable_decoder=True):
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

        self.prompt = SAMPromptEncoder(sam)

        self.decoder = SAMClassificationDecoder(
            sam=sam,
            k_layers=k_decoder_layers,
            trainable=trainable_decoder
        )
        self.geo = GeometryEnhancer(d_model=256)
        self.head = FormationHead(
            d_model=256,
            n_classes=n_classes
        )

    def forward(self, image, points_xy, points_label, valid_mask=None):
        image_embeddings = self.backbone(image)

        image_pe = self.prompt.prompt_encoder.get_dense_pe()

        sparse_pe, _ = self.prompt(
            points=(points_xy, points_label),
            boxes=None,
            masks=None
        )

        N = points_xy.shape[1]
        tokens = sparse_pe[:, :N, :] 

        tokens = self.decoder(
            tokens,
            image_embeddings.detach(),
            image_pe
        )

        tokens = self.geo(tokens, points_xy, image.shape[-2:])

        return self.head(tokens, valid_mask)
