# 🏈 Football Player Dataset Preparation & Segmentation Pipeline

This pipeline prepares, filters, labels, segments, and reviews a **football player tracking dataset** derived from **[Football Presnap Tracker (Roboflow)](https://universe.roboflow.com/football-tracking/football-presnap-tracker/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)**.  
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

| Key   | Action                                        |
| ----- | --------------------------------------------- |
| **K** | Keep image (include in dataset)               |
| **D** | Delete image (exclude from dataset)           |
| **Q** | Quit early (skip remaining images in a split) |

### 📦 Output

- Merged dataset under `merged_dataset/`
  - `images/` – consolidated image folder
  - `_annotations_merged.coco.json` – merged COCO file with all kept annotations

---

## 🥈 Step 2 – Automatic Image Orientation Correction

**Script:** `auto_image_flip.py`

### 🎯 Purpose

Ensure that every image in the dataset has the **offensive team facing the same direction** by automatically flipping images horizontally when needed.

### ⚙️ Description

- Scans each COCO annotation file in the `train`, `valid`, and `test` splits.
- Detects **quarterbacks (QB)** (`category_id == 3`) and **defensive players** (`category_id == 1`).
- Compares the x-axis center positions of bounding boxes:
  - If most QBs are **to the right** of defensive players, the image is flipped.
- Both the **image** and **bounding boxes** are flipped.
- The updated dataset is merged into a single folder containing originals and flipped versions.

### 🥈 Step 2a – Usage Example

1. Navigate to the folder containing `auto_image_flip.py`.
2. Ensure your merged dataset from Step 1 exists.
3. Run the script to process all splits (`train`, `valid`, `test`).
4. The script will automatically flip images if needed and update bounding boxes.

**Expected Output:**

- Unified folder (e.g., `Football_Presnap_Merged/`) containing:
  - Original images
  - Flipped images when needed
  - Updated COCO JSON with corrected bounding boxes

---

## 🥉 Step 3 – Classify Image Roles

**Script:** `classify_image_roles.py`

### 🎯 Purpose

Assign each image with:

- **Team roles** (offense or defense)
- **Formation types** (e.g., shotgun, singleback, etc.)
- **Field orientation**
- Automatically **blur referees and unknown players**

### ⚙️ Description

- The user identifies whether **category 1** represents the **offensive** or **defensive** team.
- The opposite role is then automatically assigned to **category 3**.
- The user selects the **offensive formation** type.
- Images can still be manually flipped if alignment needs adjustment.

### 🔑 Controls

| Key               | Action                                                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **A**             | Set category 1 as **offense first**                                                                                                 |
| **D**             | Set category 1 as **defense first**                                                                                                 |
| **1–7**           | Select formation type:<br>1️⃣ shotgun<br>2️⃣ i-formation<br>3️⃣ singleback<br>4️⃣ trips-right<br>5️⃣ trips-left<br>6️⃣ empty<br>7️⃣ pistol |
| **Enter / Space** | Confirm classification                                                                                                              |
| **F**             | Flip image horizontally                                                                                                             |
| **U**             | Unblur referees and unknown players                                                                                                 |
| **S**             | Skip current image                                                                                                                  |
| **Q**             | Quit early                                                                                                                          |

### 📦 Output

- Updated annotations with consistent team roles, formations, and field orientation

---

## 🏅 Step 4 – Interactive Additional Player Information Labeling

**Script:** `additional_info_labeling.py`

### 🎯 Purpose

Automatically supports adding **multiple types of additional information** to each player:

- **Position** (RB, WR, TE, etc.)
- **Alignment** (Left, Right, Center)
- **Any custom labels** you define

This step is **interactive** but updates the annotations **automatically** with each key press.

### ⚙️ Description

1. Load the merged COCO dataset with images and existing annotations.
2. For each player annotation:
   - Draw the bounding box and current labels on the image.
   - Display **instructions** for assigning labels with keys.
   - Update the annotation automatically based on the pressed key.
3. Repeat until all players are labeled or skipped.
4. Supports **any number of additional information types** through a configurable dictionary (e.g., `position`, `alignment`, or custom labels).

### 🔑 Controls

| Key   | Action                                     |
| ----- | ------------------------------------------ |
| 1–3   | Assign **Position** (RB, WR, TE, etc.)     |
| R/L/C | Assign **Alignment** (Right, Left, Center) |
| S     | Skip current player                        |
| Enter | Confirm player with current labels         |
| Q     | Quit labeling process                      |

### 📦 Output

- Updated COCO JSON `_annotations_additional_info.coco.json` containing all additional information for each player
- Optional visual confirmation of assigned labels

---

## 🏆 Step 5 – Automatic Segmentation with SAM

**Script:** `auto_segment_sam.py`

### 🎯 Purpose

Automatically generate segmentation masks for each player bounding box using **Meta’s Segment Anything Model (SAM)**.

### ⚙️ Description

- Loads the merged COCO dataset with additional player information.
- Applies SAM’s segmentation guided by player bounding boxes.
- Produces accurate **binary masks** for each player automatically.
- Saves each mask as a `.png` file and links it to its corresponding annotation in the COCO file.

### 📦 Output

- Auto-generated masks in `auto_masks/`
- Updated COCO file `_annotations_roles_masks_auto.coco.json`

---

## 🏆 Step 6 – Review and Re-Segment Masks

**Script:** `review_and_resegment.py`

### 🎯 Purpose

Visually review and manually correct segmentation masks using **interactive SAM refinement** or **manual painting**.

### ⚙️ Description

- Opens an interactive viewer with bounding boxes and current masks.
- Allows re-running SAM with positive/negative points or painting directly.
- Saves refined masks and updates the COCO JSON instantly.

### 📦 Output

- Verified and corrected masks
- Final COCO annotation file with high-quality segmentation

---

### ✅ End Result

A fully processed, direction-aligned, annotated, and segmented football dataset ready for:

- Formation and strategy analysis
- Pose estimation or tracking tasks
- Model training (segmentation, detection, or action recognition)
