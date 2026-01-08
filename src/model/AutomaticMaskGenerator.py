import os
import sys
import urllib.request
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class AutomaticMaskGenerator:
    def __init__(self, sam_type="vit_h", ckpt_dir="./model/models", unfreeze_last_blocks=0):
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

        self.sam = sam_model_registry[sam_type](checkpoint=ckpt_path)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,            # controls granularity
            pred_iou_thresh=0.88,
            stability_score_thresh=0.9,
            min_mask_region_area=500,      # important for small players
        )

    def generate_masks(self, image):
        masks = self.mask_generator.generate(image)
        return masks