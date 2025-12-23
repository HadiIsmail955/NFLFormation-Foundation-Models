import os
import json

source_dir = r".\Data\data_convertor_Football_Presnap_Tracker\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_additional_info.coco.json"

images_dir = os.path.join(source_dir, "images")
auto_masks_dir = os.path.join(source_dir, "auto_masks")

team_masks_root = os.path.join(source_dir, "Team_masks")
all_team_dir = os.path.join(team_masks_root, "all_masks")
off_team_dir = os.path.join(team_masks_root, "off_masks")
def_team_dir = os.path.join(team_masks_root, "def_masks")


coco_path = os.path.join(source_dir, coco_file)
with open(coco_path, "r") as f:
    coco = json.load(f)

    # alignments = {
    #     ann["alignment"]
    #     for ann in coco.get("annotations", [])
    #     if "alignment" in ann
    # }
    # positions = {
    #     ann["position"]
    #     for ann in coco.get("annotations", [])
    #     if "position" in ann
    # }

    # print("alignments", sorted(alignments))
    # print("positions", sorted(positions))



def check_folder(expected_set, folder_path, label):
    valid_exts = {".png"}
    folder_files = {
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    }

    missing = sorted(expected_set - folder_files)
    extra = sorted(folder_files - expected_set)

    print(f"\n------- {label} -------")
    print(f"Expected masks : {len(expected_set)}")
    print(f"Files in folder: {len(folder_files)}\n")

    if missing:
        print(f"Missing {label.lower()} ({len(missing)}):")
        for m in missing:
            print("  -", m)
    else:
        print(f"No missing {label.lower()}!")

    if extra:
        print(f"\nExtra {label.lower()} ({len(extra)}):")
        for m in extra:
            print("  -", m)
    else:
        print(f"No extra files in {label.lower()}!")

    return missing, extra

coco_images = {img["file_name"] for img in coco["images"]}
valid_img_exts = {".jpg", ".jpeg", ".png"}

folder_images = {
    f for f in os.listdir(images_dir)
    if os.path.splitext(f)[1].lower() in valid_img_exts
}

missing_images = sorted(coco_images - folder_images)
extra_images   = sorted(folder_images - coco_images)


print("\n==========================")
print("COCO DATA VERIFICATION")
print("==========================\n")

print("------- IMAGES -------")
print(f"Images in JSON  : {len(coco_images)}")
print(f"Images in folder: {len(folder_images)}\n")

if missing_images:
    print(f"Missing images ({len(missing_images)}):")
    for img in missing_images:
        print("  -", img)
else:
    print("No missing images!")

if extra_images:
    print(f"\nExtra images ({len(extra_images)}):")
    for img in extra_images:
        print("  -", img)
else:
    print("No extra images!")

# ===========================
auto_masks_expected = {
    ann["segmentation_mask"]
    for ann in coco["annotations"]
    if "segmentation_mask" in ann
}

team_masks_expected = {
    img["team_mask"]
    for img in coco["images"]
}

off_masks_expected = {
    img["offense_mask"]
    for img in coco["images"]
}

def_masks_expected = {
    img["defense_mask"]
    for img in coco["images"]
    }

check_folder(auto_masks_expected, auto_masks_dir, "AUTO MASKS")
check_folder(team_masks_expected, all_team_dir, "TEAM MASKS")
check_folder(off_masks_expected, off_team_dir, "OFFENSE MASKS")
check_folder(def_masks_expected, def_team_dir, "DEFENSE MASKS")

print("\nVerification complete.\n")
