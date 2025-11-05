import os, cv2, json
from tqdm import tqdm

# === CONFIGURATION ===
merged_coco_path = "_annotations_all.coco.json"
images_dir = "images"
output_json = "_annotations_roles.coco.json"
source_file = r"..\Football Jersey Tracker.v1i.coco\merged_dataset"
labeled_dir = os.path.join(source_file, "labeled_images")
os.makedirs(labeled_dir, exist_ok=True)

blur_unknown = True
UNBLUR_KEY = ord("u")  # Press 'U' to save unblurred

# === KEY MAPPING ===
ATTACK_FIRST_KEY = ord("a")
DEFENSE_FIRST_KEY = ord("d")
SKIP_KEY = ord("s")
QUIT_KEY = ord("q")
CONFIRM_KEYS = [13, 32]  # Enter or Space

formations = {
    ord("1"): "shotgun",
    ord("2"): "i-formation",
    ord("3"): "singleback",
    ord("4"): "trips-right",
    ord("5"): "trips-left",
    ord("6"): "empty",
    ord("7"): "pistol"
}

# === LOAD COCO ===
with open(os.path.join(source_file, merged_coco_path), "r") as f:
    coco = json.load(f)

annotations = coco["annotations"]
images = coco["images"]

# build annotation index
anns_by_image = {}
for ann in annotations:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

print("🎯 Starting Visual Labeling Tool with Save & Unblur Option")

updated_images = []

for img in tqdm(images):
    img_path = os.path.join(source_file, images_dir, img["file_name"])
    if not os.path.exists(img_path):
        continue

    anns = anns_by_image.get(img["id"], [])
    original_image = cv2.imread(img_path)
    if original_image is None:
        continue

    # Prepare display and save images
    display_image = original_image.copy()  # for showing bounding boxes
    save_image = original_image.copy()     # for saving (blurred or unblurred)
    
    # Pre-blur unknown/referee for display and save_image
    if blur_unknown:
        for ann in anns:
            x, y, w, h = map(int, ann["bbox"])
            cat = ann["category_id"]
            if cat not in [1, 3]:  # unknown/referee
                roi = save_image[y:y+h, x:x+w]
                save_image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (23,23), 30)
    
    # Draw bounding boxes only for display
    for ann in anns:
        x, y, w, h = map(int, ann["bbox"])
        cat = ann["category_id"]
        if cat == 1:
            color = (0, 255, 0)
        elif cat == 3:
            color = (255, 0, 0)
        else:
            color = (128, 128, 128)
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display_image, f"cat:{cat}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    attack_cat, defense_cat, formation = None, None, None
    confirmed = False
    unblur = False

    while True:
        disp = display_image.copy()

        # Overlay instructions
        y0 = 25
        instructions = [
            "A: attack first | D: defense first",
            "1–7: select formation | S: skip | Q: quit | U: unblur save",
            "Unknown/referee are gray (blurred by default)"
        ]
        for line in instructions:
            cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y0 += 30

        # Overlay formation options
        y_form = y0
        cv2.putText(disp, "Formations:", (10, y_form), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y_form += 30
        for k, v in formations.items():
            text = f"{chr(k)}: {v}"
            cv2.putText(disp, text, (20, y_form), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
            y_form += 25

        # Show current selection
        status = f"Attack: {attack_cat or '-'} | Defense: {defense_cat or '-'} | Formation: {formation or '-'}"
        cv2.putText(disp, status, (10, y_form + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Football Labeling Tool", disp)
        key = cv2.waitKey(0)

        if key == QUIT_KEY:
            print("🛑 Exiting early.")
            cv2.destroyAllWindows()
            exit()
        elif key == SKIP_KEY:
            print("⏭️ Skipping image.")
            break
        elif key == ATTACK_FIRST_KEY:
            attack_cat, defense_cat = 1, 3
            print("⚽ Attack team = category 1, Defense = category 3")
        elif key == DEFENSE_FIRST_KEY:
            attack_cat, defense_cat = 3, 1
            print("🛡️ Defense team = category 1, Attack = category 3")
        elif key in formations:
            formation = formations[key]
            print(f"📋 Formation selected: {formation}")
        elif key in CONFIRM_KEYS:
            if attack_cat and formation:
                confirmed = True
                break
            else:
                print("❗ Please select team roles and formation before confirming.")
        elif key == UNBLUR_KEY:
            unblur = True
            print("🔓 Unblur activated for saving this image")

    if confirmed:
        img["attributes"] = {
            "formation": formation,
            "attack_team_category": attack_cat,
            "defense_team_category": defense_cat
        }
        updated_images.append(img)

        # Save labeled image
        save_final = original_image.copy() if unblur else save_image
        save_path = os.path.join(labeled_dir, img["file_name"])
        cv2.imwrite(save_path, save_final)

cv2.destroyAllWindows()

# Save updated COCO
coco["images"] = updated_images
with open(os.path.join(source_file, output_json), "w") as f:
    json.dump(coco, f, indent=2)

print(f"\n✅ Saved labeled dataset with roles and formations to: {output_json}")
print(f"✅ Labeled images saved to: {labeled_dir}")
