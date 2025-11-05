import os, json, cv2, numpy as np, torch, urllib.request
from segment_anything import SamPredictor, sam_model_registry

# === CONFIGURATION ===
source_dir = r"..\Football Jersey Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_roles_masks_auto.coco.json"
images_dir = os.path.join(source_dir, "images")
masks_dir = os.path.join(source_dir, "auto_masks")
output_json = "_annotations_roles_masks_resegmented.coco.json"

# === MODEL SETUP ===
model_type = "vit_h"
sam_checkpoint = f"sam_{model_type}.pth"
if not os.path.exists(sam_checkpoint):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    print("⚡ Downloading SAM model...")
    urllib.request.urlretrieve(url, sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# === LOAD COCO ===
with open(os.path.join(source_dir, coco_file)) as f:
    coco = json.load(f)
images = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]

# === SAM SEGMENT FUNCTION ===
def segment_box(image, bbox):
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x, y, w, h = map(int, bbox)
    masks, scores, _ = predictor.predict(box=np.array([x, y, x+w, y+h]), multimask_output=True)
    return masks[np.argmax(scores)].astype(np.uint8)

# === REVIEW MODE ===
KEEP_KEY, RESEG_KEY, QUIT_KEY = ord("k"), ord("r"), ord("q")
print("🎯 Review masks: K=keep, R=resegment, Q=quit early")

reviewed = []

for ann in anns:
    mask_file = ann.get("segmentation_mask")
    if not mask_file:
        continue

    img_info = images[ann["image_id"]]
    img_path = os.path.join(images_dir, img_info["file_name"])
    mask_path = os.path.join(masks_dir, mask_file)
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue
    mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None

    if mask is not None:
        overlay = image.copy()
        overlay[mask > 0] = (0, 255, 0)
        preview = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    else:
        preview = image.copy()

    cv2.imshow("Review (K=keep, R=resegment, Q=quit)", preview)
    key = cv2.waitKey(0)
    if key == QUIT_KEY:
        print("🛑 Quit early.")
        break

    if key == RESEG_KEY:
        print("📏 Draw new bounding box and press ENTER.")
        r = cv2.selectROI("Draw new box", image, fromCenter=False, showCrosshair=True)
        if r != (0,0,0,0):
            print("🔁 Re-segmenting ...")
            new_mask = segment_box(image, r)
            cv2.imwrite(mask_path, new_mask * 255)
            ann["bbox"] = [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
            print("✅ Mask updated.")
        cv2.destroyWindow("Draw new box")

    reviewed.append(ann)

cv2.destroyAllWindows()
coco["annotations"] = reviewed
out_path = os.path.join(source_dir, output_json)
with open(out_path, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ Re-segmented dataset saved to {out_path}")