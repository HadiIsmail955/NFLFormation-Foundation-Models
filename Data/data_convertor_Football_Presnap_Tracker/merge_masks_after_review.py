import os, sys, json, cv2, torch, numpy as np, urllib.request
from tqdm import tqdm
from collections import defaultdict
from utils.merge_masks import merge_team_masks, merge_team_masks_color

# === CONFIGURATION ===
source_dir = r".\Football Presnap Tracker.v1i.coco\merged_dataset"
coco_file = "_annotations_masks_auto.coco.json"
masks_dir = os.path.join(source_dir, "auto_masks")
team_masks_dir = os.path.join(source_dir, "Team_masks")
os.makedirs(team_masks_dir, exist_ok=True)
off_team_masks_dir = os.path.join(team_masks_dir, "off_masks")
os.makedirs(off_team_masks_dir, exist_ok=True)
def_team_masks_dir = os.path.join(team_masks_dir, "def_masks")
os.makedirs(def_team_masks_dir, exist_ok=True)
all_team_masks_dir = os.path.join(team_masks_dir, "all_masks")
os.makedirs(all_team_masks_dir, exist_ok=True)
output_json = "_annotations_mergered_masks_auto.coco.json"

# === LOAD COCO DATA ===
with open(os.path.join(source_dir, coco_file)) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
anns = coco["annotations"]

# === GROUP ANNOTATIONS PER IMAGE ===
anns_by_image = defaultdict(list)
for ann in anns:
    anns_by_image[ann["image_id"]].append(ann)
    
# === MERGE TEAM MASKS AFTER REVIEW ===
for image_id, ann_list in tqdm(anns_by_image.items()):
    team_masks = {"offense": [], "defense": []}
    img_info = images[image_id]
    for ann in ann_list:
        cat = ann["category_id"]
        if cat in [2,3,5]:  # Offense
            team_masks["offense"].append(os.path.join(masks_dir, ann["segmentation_mask"]))
        elif cat in [1]:  # Defense
            team_masks["defense"].append(os.path.join(masks_dir, ann["segmentation_mask"]))
    
    team_mask_paths = os.path.join(all_team_masks_dir, f"{image_id}_team_mask.png")
    off_team_masks_paths = os.path.join(off_team_masks_dir, f"{image_id}_off_mask.png")
    def_team_masks_paths = os.path.join(def_team_masks_dir, f"{image_id}_def_mask.png")

    merge_team_masks(team_masks["offense"], output_path=off_team_masks_paths)
    merge_team_masks(team_masks["defense"], output_path=def_team_masks_paths)

    merge_team_masks_color(
        offense_mask_path=off_team_masks_paths,
        defense_mask_path=def_team_masks_paths,
        output_path=team_mask_paths
    )
    # Update annotation to point to new team mask
    images[image_id]["team_mask"] = os.path.relpath(team_mask_paths, source_dir)
    images[image_id]["offense_mask"] = os.path.relpath(off_team_masks_paths, source_dir)
    images[image_id]["defense_mask"] = os.path.relpath(def_team_masks_paths, source_dir)
    
# === SAVE UPDATED COCO JSON ===
with open(os.path.join(source_dir, output_json), 'w') as f:
    json.dump(coco, f)
print("Merged team masks saved and COCO JSON updated.")
    

        