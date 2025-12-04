import os
import torch
import numpy as np
from PIL import Image
from model.backbone.SAMBackbone import SAMBackbone
from model.preproccesor.SAMPreprocessor import SAMPreprocessor
from util.resize import reverse_resize_and_pad

data_folder = "./Data/data_convertor_Football_Presnap_Tracker/Football Presnap Tracker.v1i.coco/merged_dataset/"
image_folder = "resize_images"
mask_folder="Team_masks/off_masks"

image_file="1_PNG_jpg.rf.06f486666e063bcd3c153585d71f9085.jpg"
mask_file="1_PNG_jpg.rf.06f486666e063bcd3c153585d71f9085.jpg_off_mask.png"

image_path = os.path.join(data_folder, image_folder, image_file)
mask_path = os.path.join(data_folder, mask_folder, mask_file)

# Load images
image = np.array(Image.open(image_path).convert("RGB"))
print(f"before image shape {image.shape}")

# Preprocess
preprocessor = SAMPreprocessor(target_size=1024)
image_pre = preprocessor(image)
print(f"after image shape {image_pre.shape}")

# Create SAM backbone
sam_backbone = SAMBackbone(sam_type="vit_h")

# Forward pass
with torch.no_grad():
    features = sam_backbone(image_pre)

print("Embedding shape:", features.shape)

meta= {
                "orig_h": 1079,
                "orig_w": 1920,
                "new_h": 575,
                "new_w": 1024,
                "pad_top": 224,
                "pad_bottom": 225,
                "pad_left": 0,
                "pad_right": 0,
                "scale": 0.5333333333333333
            }
rev_image=reverse_resize_and_pad(image,meta)
print("reverse ", rev_image.shape)

import matplotlib.pyplot as plt

def show_tensor_image(t):
    """Undo SAM normalization and display."""
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1,3,1,1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1,3,1,1)

    # unnormalize → move to CPU → convert to uint8
    img = (t * pixel_std + pixel_mean)[0].permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)

    plt.imshow(img)
    plt.axis("off")


plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,2,1)
plt.title("rev Image")
plt.imshow(rev_image)
plt.axis("off")

# processed
plt.subplot(1,2,2)
plt.title("Preprocessed (1024×1024 padded)")
show_tensor_image(image_pre)

plt.show()
