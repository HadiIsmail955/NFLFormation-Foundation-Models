import json
import os
import cv2
import shutil
from tqdm import tqdm

# === CONFIGURATION ===
source_dir = "../Football Jersey Tracker.v1i.coco"
splits = ["train", "valid", "test"]
input_json_name = "_annotations.coco.json"
filtered_json_name = "_annotations_filtered.coco.json"

merged_output_dir = os.path.join(source_dir, "merged_dataset")
merged_images_dir = os.path.join(merged_output_dir, "images")
merged_json_path = os.path.join(merged_output_dir, "_annotations_all.coco.json")

os.makedirs(merged_images_dir, exist_ok=True)

# === KEY CONTROLS ===
KEEP_KEY = ord("k")  # keep
DELETE_KEY = ord("d")  # delete
QUIT_KEY = ord("q")  # quit early

# === PREPARE STORAGE ===
all_images = []
all_annotations = []
categories = None
image_id_offset = 0
annotation_id_offset = 0

# === PROCESS EACH SPLIT ===
for split in splits:
    print(f"\n📁 Processing split: {split}")
    split_dir = os.path.join(source_dir, split)
    coco_path = os.path.join(split_dir, input_json_name)

    if not os.path.exists(coco_path):
        print(f"⚠️ Missing: {coco_path}, skipping this split.")
        continue

    # --- Load JSON ---
    with open(coco_path, "r") as f:
        coco = json.load(f)

    if categories is None:
        categories = coco["categories"]

    images = coco["images"]
    annotations = coco.get("annotations", [])
    kept_images = []
    kept_ids = set()

    print(f"🔍 Loaded {len(images)} images for review...")

    # --- Loop through images ---
    for img in tqdm(images, desc=f"Reviewing {split} images"):
        img_path = os.path.join(split_dir, img["file_name"])
        if not os.path.exists(img_path):
            print(f"⚠️ Missing file: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Cannot read: {img_path}")
            continue

        # Show image
        cv2.imshow("Review (k=keep, d=delete, q=quit)", image)
        key = cv2.waitKey(0)

        if key == QUIT_KEY:
            print("🛑 Exiting early review.")
            break
        elif key == KEEP_KEY:
            kept_images.append(img)
            kept_ids.add(img["id"])
            print(f"✅ Kept: {img['file_name']}")
        elif key == DELETE_KEY:
            print(f"❌ Deleted: {img['file_name']}")
        else:
            print("⏭️ Unknown key pressed — skipping this image.")

    cv2.destroyAllWindows()

    # --- Filter annotations ---
    filtered_annotations = [ann for ann in annotations if ann["image_id"] in kept_ids]

    # --- Save filtered JSON for this split ---
    filtered_coco = {
        **coco,
        "images": kept_images,
        "annotations": filtered_annotations,
    }
    filtered_json_path = os.path.join(split_dir, filtered_json_name)
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_coco, f, indent=2)

    print(f"💾 Saved filtered annotations to {filtered_json_path}")
    print(f"🖼️ Kept {len(kept_images)} / {len(images)} images")

    # --- Add to merged dataset ---
    for img in kept_images:
        new_id = img["id"] + image_id_offset
        old_id = img["id"]
        img["id"] = new_id

        # Copy image to merged directory
        src_path = os.path.join(split_dir, img["file_name"])
        dst_path = os.path.join(merged_images_dir, img["file_name"])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    for ann in filtered_annotations:
        ann["id"] += annotation_id_offset
        ann["image_id"] += image_id_offset

    all_images.extend(kept_images)
    all_annotations.extend(filtered_annotations)

    if all_images:
        image_id_offset = max(img["id"] for img in all_images) + 1
    if all_annotations:
        annotation_id_offset = max(ann["id"] for ann in all_annotations) + 1

# === SAVE MERGED DATASET ===
merged_coco = {
    "info": {
        "description": "Merged & filtered dataset (train+valid+test)",
        "version": "1.0",
    },
    "licenses": [],
    "categories": categories,
    "images": all_images,
    "annotations": all_annotations,
}

with open(merged_json_path, "w") as f:
    json.dump(merged_coco, f, indent=2)

print(f"\n✅ Merged {len(all_images)} images and {len(all_annotations)} annotations.")
print(f"💾 Saved merged dataset JSON to {merged_json_path}")
print(f"📁 Merged images are stored in: {merged_images_dir}")
