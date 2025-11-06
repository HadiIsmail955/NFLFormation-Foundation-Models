# 🏈 Football Player Dataset Preparation & Segmentation Pipeline

This pipeline prepares, filters, labels, segments, and reviews a **football player tracking dataset** derived from **[Football Jersey Tracker (Roboflow)](https://universe.roboflow.com/football-tracking/football-jersey-tracker/browse?queryText=&pageSize=50&startingIndex=50&browseQuery=true)**.  
It standardizes images, player annotations, team roles, and segmentation masks for further training or analysis.

---

## 🥇 Step 1 – Filter and Merge Dataset

**Script:** `filter_and_merge_coco.py`

### 🎯 Purpose
Select only the relevant and high-quality images from the dataset (train/valid/test) and merge them into a single balanced dataset.

### ⚙️ Description
- The user manually reviews each image and decides whether to **keep** or **discard** it.
- After the review, the script automatically merges all **kept** images and annotations into one unified dataset for easier processing.

### 🔑 Controls

| Key | Action |
|-----|--------|
| **K** | Keep image (include in dataset) |
| **D** | Delete image (exclude from dataset) |
| **Q** | Quit early (skip remaining images in a split) |

> 🧩 The script processes images from **train**, **validation**, and **test** splits sequentially.

### 📦 Output
- Merged dataset under `merged_dataset/`
  - `images/` – consolidated image folder  
  - `_annotations_merged.coco.json` – merged COCO file with all kept annotations

---

## 🥈 Step 2 – Classify Image Roles

**Script:** `classify_image_roles.py`

### 🎯 Purpose
Assign each image with:
- **Team roles** (offense or defense)
- **Formation types** (e.g., shotgun, singleback, etc.)
- **Field orientation** (ensure the offensive side is consistent)
- Automatically **blur referees and unknown players**

### ⚙️ Description
- The user identifies whether **category 1** represents the **offensive team** or **defensive team**.  
  The opposite role is then automatically assigned to **category 3**.
- The user then selects the **offensive formation** type.
- The image can be **flipped horizontally** to align all offensive teams to one side of the field.

### 🔑 Controls

| Key | Action |
|-----|--------|
| **A** | Set category 1 as **offense first** |
| **D** | Set category 1 as **defense first** |
| **1–7** | Select formation type:<br>1️⃣ shotgun<br>2️⃣ i-formation<br>3️⃣ singleback<br>4️⃣ trips-right<br>5️⃣ trips-left<br>6️⃣ empty<br>7️⃣ pistol |
| **Enter / Space** | Confirm classification |
| **F** | Flip image horizontally |
| **U** | Unblur referees and unknown players |
| **S** | Skip current image |
| **Q** | Quit early |

### 📦 Output
- Updated annotations with team roles, formations, and consistent image orientation.

---

## 🥉 Step 3 – Automatic Segmentation with SAM

**Script:** `auto_segment_sam.py`

### 🎯 Purpose
Automatically generate segmentation masks for each player bounding box using **Meta’s Segment Anything Model (SAM)**.

### ⚙️ Description
- Loads the merged COCO dataset and applies SAM’s segmentation guided by player bounding boxes.
- Produces accurate **binary masks** for each player automatically.
- Saves each mask as a `.png` file and links it to the corresponding annotation in the COCO file.

### 📦 Output
- Auto-generated player masks saved in `auto_masks/`
- Updated COCO annotation file (e.g. `_annotations_roles_masks_auto.coco.json`) containing references to mask files

---

## 🏆 Step 4 – Review and Re-Segment Masks

**Script:** `review_and_resegment.py`

### 🎯 Purpose
Visually review and manually correct segmentation masks for each player — using either **interactive SAM refinement** or **manual mask painting**.

### ⚙️ Description
- Opens an interactive image viewer.
- Displays each image with bounding boxes and existing segmentation overlays.
- Click on any bounding box to enter **edit mode** for that player.
- You can either:
  1. **Re-run SAM segmentation** with positive/negative point prompts, or  
  2. **Manually paint/erase** mask regions pixel-by-pixel.

### 🧠 Features
- Combine SAM’s automatic segmentation with precise manual correction.
- Update and save refined masks instantly.
- All changes update the COCO annotation file and mask images automatically.

### 🔑 Controls

| Key | Action |
|-----|--------|
| **n / p** | Next / previous image |
| **q** | Quit program |
| **Click on a bbox** | Enter edit mode for that player |
| **Left-click** | Add positive point for SAM |
| **Right-click** | Add negative point for SAM |
| **Middle-click** | Clear current points |
| **r** | Re-run SAM re-segmentation using current points |
| **s** | Save current (auto or manual) mask |
| **e** | Exit edit mode |
| **u** | Undo last point |
| **m** | Toggle **manual paint mode** |
| **Left-drag (paint mode)** | Paint (add mask region) |
| **Right-drag (paint mode)** | Erase (remove mask region) |

### 💾 Behavior
- Edited masks are saved to `auto_masks/` as `.png` files.
- The COCO JSON file is updated to link each annotation to its final verified mask.

### 📦 Output
- Fully verified, corrected segmentation masks
- Final COCO annotation file containing accurate player segmentation data

---

## 📚 Summary of Workflow

| Step | Script | Purpose |
|------|---------|----------|
| **1** | `filter_and_merge_coco.py` | Filter and merge dataset splits into one |
| **2** | `classify_image_roles.py` | Assign team roles, formations, and orientations |
| **3** | `auto_segment_sam.py` | Auto-segment players using SAM |
| **4** | `review_and_resegment.py` | Review and refine segmentations manually |

---

### ✅ End Result
A fully processed, annotated, and segmented football dataset ready for:
- Team and formation analysis  
- Pose estimation or tracking tasks  
- Model training (e.g. segmentation or action recognition)
