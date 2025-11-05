import json
import os
import cv2
from tqdm import tqdm

# === CONFIGURATION ===
source_dir="..\Football Jersey Tracker.v1i.coco"
dirNames=["train","valid","test"]
coco_path = "_annotations.coco.json"
output_json = "_annotations_filtered.coco.json"

# === KEY CONTROLS ===
KEEP_KEY = ord("k")  # press "k" to keep
DELETE_KEY = ord("d")  # press "d" to delete/skip
QUIT_KEY = ord("q")  # press "q" to quit early

for dirName in dirNames:
    # === LOAD COCO DATA ===
    input_file = os.path.join(source_dir, dirName, coco_path)
    with open(input_file, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco.get("annotations", [])
    kept_images = []
    kept_ids = set()

    print(f"🔍 Loaded {len(images)} images. Starting review...")

    # === LOOP THROUGH IMAGES ===
    for img in tqdm(images, desc="Reviewing images"):
        file_path = os.path.join(source_dir, dirName, img["file_name"])
        if not os.path.exists(file_path):
            print(f"⚠️ Missing: {file_path}")
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"⚠️ Could not read: {file_path}")
            continue

        # Display image
        cv2.imshow("Review (k=keep, d=delete, q=quit)", image)
        key = cv2.waitKey(0)

        if key == QUIT_KEY:
            print("🛑 Exiting early.")
            break
        elif key == KEEP_KEY:
            kept_images.append(img)
            kept_ids.add(img["id"])
            print(f"✅ Kept: {img['file_name']}")
        elif key == DELETE_KEY:
            print(f"❌ Deleted: {img['file_name']}")
        else:
            print(f"⏭️ Skipped: {img['file_name']} (press k/d/q next time)")

    cv2.destroyAllWindows()

    # === FILTER ANNOTATIONS ===
    filtered_annotations = [ann for ann in annotations if ann["image_id"] in kept_ids]

    filtered_coco = {
        **coco,
        "images": kept_images,
        "annotations": filtered_annotations,
    }

    # === SAVE OUTPUT ===
    output_file = os.path.join(source_dir, dirName, output_json)
    with open(output_file, "w") as f:
        json.dump(filtered_coco, f, indent=2)

    print(f"\n✅ Saved filtered dataset: {output_json}")
    print(f"🖼️ Kept {len(kept_images)} images (out of {len(images)})")
