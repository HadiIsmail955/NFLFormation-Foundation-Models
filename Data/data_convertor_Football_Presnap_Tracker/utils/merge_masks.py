import cv2
import numpy as np
import os

def merge_team_masks(mask_files, output_path=None):
    merged = None

    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = (mask > 128).astype(np.uint8)

        if merged is None:
            merged = mask
        else:
            merged = np.maximum(merged, mask) 

    if merged is not None and output_path is not None:
        cv2.imwrite(output_path, merged * 255)
        # print("Saved merged mask:", output_path)


def merge_team_masks_color(offense_mask_path, defense_mask_path, output_path=None,
                           offense_color=(255, 0, 0),
                           defense_color=(0, 0, 255),
                           overlap_color=None):

    offense = cv2.imread(offense_mask_path, cv2.IMREAD_GRAYSCALE)
    defense = cv2.imread(defense_mask_path, cv2.IMREAD_GRAYSCALE)

    if offense is None or defense is None:
        raise ValueError("One of the team mask files could not be read.")

    offense = (offense > 128).astype(np.uint8)
    defense = (defense > 128).astype(np.uint8)
    h, w = offense.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    overlap = (offense & defense).astype(bool)
    color_mask[defense == 1] = defense_color
    color_mask[offense == 1] = offense_color

    if overlap_color is not None:
        color_mask[overlap] = overlap_color

    if output_path is not None:
        cv2.imwrite(output_path, color_mask)
        # print(f"Saved merged color team mask: {output_path}")