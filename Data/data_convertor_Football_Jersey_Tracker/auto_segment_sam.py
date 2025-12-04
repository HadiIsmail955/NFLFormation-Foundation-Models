# !pip install git+https://github.com/facebookresearch/segment-anything.git
# !pip install opencv-python matplotlib torch torchvision pycocotools
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# sam_vit_h_4b8939.pth
# sam_vit_l_0b3195.pth
# sam_vit_b_01ec64.pth
import os, sys, json, cv2, torch, numpy as np, urllib.request
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry

# === CONFIGURATION ===
source_dir = r".\Football Jersey Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_roles.coco.json"
images_dir = os.path.join(source_dir, "images")
masks_dir = os.path.join(source_dir, "auto_masks")
os.makedirs(masks_dir, exist_ok=True)
output_json = "_annotations_roles_masks_auto.coco.json"

# === MODEL SELECTION ===
model_choice = input("Select SAM model (h=vit_h, l=vit_l, b=vit_b) [default h]: ").lower()
model_type = {"h": "vit_h", "l": "vit_l", "b": "vit_b"}.get(model_choice, "vit_h")
device = "cuda" if torch.cuda.is_available() else "cpu"

# === SAM CHECKPOINTS ===
sam_urls = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
sam_checkpoint = f"sam_{model_type}.pth"

# === DOWNLOAD SAM IF NEEDED ===
if not os.path.exists(sam_checkpoint):
    url = sam_urls[model_type]
    print(f"⚡ Downloading SAM checkpoint ({model_type}) from {url} ...")
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        progress = min(int(downloaded / total_size * 100), 100)
        sys.stdout.write(f"\rDownloading: {progress}%")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, sam_checkpoint, reporthook=show_progress)
    print("\n✅ Download complete!")

print(f"🚀 Using SAM {model_type} on {device}")

# === LOAD SAM MODEL ===
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# === LOAD COCO DATA ===
with open(os.path.join(source_dir, coco_file)) as f:
    coco = json.load(f)
images = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]

# === SEGMENT FUNCTION ===
def segment_box(image, bbox):
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x, y, w, h = map(int, bbox)
    masks, scores, _ = predictor.predict(box=np.array([x, y, x+w, y+h]), multimask_output=True)
    return masks[np.argmax(scores)].astype(np.uint8)

# === AUTO SEGMENTATION ===
print(f"⚡ Segmenting {len(anns)} player bounding boxes...")
for ann in tqdm(anns):
    img_info = images[ann["image_id"]]
    img_path = os.path.join(images_dir, img_info["file_name"])
    if not os.path.exists(img_path):
        continue
    image = cv2.imread(img_path)
    if image is None:
        continue

    mask = segment_box(image, ann["bbox"])
    mask_name = f"{os.path.splitext(img_info['file_name'])[0]}_{ann['id']}.png"
    cv2.imwrite(os.path.join(masks_dir, mask_name), mask * 255)
    ann["segmentation_mask"] = mask_name

print(f"✅ All auto masks saved to {masks_dir}")

# === SAVE OUTPUT ===
coco["annotations"] = anns
out_path = os.path.join(source_dir, output_json)
with open(out_path, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ Updated COCO JSON saved as {out_path}")
