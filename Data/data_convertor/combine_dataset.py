import json
import os

# === INPUT FILES ===
source_dir="..\Football Jersey Tracker.v1i.coco"
splits = ["train","valid","test"]
coco_path = "_annotations_filtered.coco.json"
output_path = "merge\_annotations_all.coco.json"

all_images = []
all_annotations = []
categories = None
image_id_offset = 0
annotation_id_offset = 0

for split_file in splits:
    split_file = os.path.join(source_dir, split_file, coco_path)
    if not os.path.exists(split_file):
        print(f"⚠️ Missing {split_file}, skipping.")
        continue

    with open(split_file, "r") as f:
        coco = json.load(f)

    if categories is None:
        categories = coco["categories"]  # assume same categories across all splits

    # --- Update image and annotation IDs to avoid duplicates ---
    images = coco["images"]
    annotations = coco["annotations"]

    for img in images:
        img["id"] += image_id_offset
    for ann in annotations:
        ann["id"] += annotation_id_offset
        ann["image_id"] += image_id_offset

    all_images.extend(images)
    all_annotations.extend(annotations)

    image_id_offset = max(img["id"] for img in all_images) + 1
    annotation_id_offset = max(ann["id"] for ann in all_annotations) + 1

print(f"✅ Merged {len(all_images)} images and {len(all_annotations)} annotations.")

# --- Create merged dataset ---
merged_coco = {
    "info": {
        "description": "Merged dataset (train+val+test)",
        "version": "1.0",
    },
    "licenses": [],
    "categories": categories,
    "images": all_images,
    "annotations": all_annotations,
}
output_path=os.path.join(source_dir, output_path)
with open(output_path, "w") as f:
    json.dump(merged_coco, f, indent=2)

print(f"💾 Saved merged dataset to {output_path}")
