import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PresnapDataset(Dataset):
    RARE_FORMATIONS = ['pistol', 'victory', 'stacked', 'twins-left']

    ROLE_MAP = {
        "oline": 0,
        "qb": 1,
        "skill": 2
    }
    
    ALIGNMENT_MAP = {
         'Left':0, 
         'Pistol':1, 
         'Right':2, 
         'Shotgun':3, 
         'Under Center':4
    }
    POSITION_MAP = {
            'FB':0,
            'QB':1,
            'RB':2, 
            'SB':3, 
            'TE':4, 
            'WB':5, 
            'WR':6
        }

    FORMATION_MAP = {
        "shotgun": 0,
        "singleback": 1,
        "ace-left": 2,
        "ace-right": 3,
        "trips-left": 4,
        "trips-right": 5,
        "twins-right": 6,
        "bunch-left": 7,
        "bunch-right": 8,
        "i-formation": 9,
        "trey-left": 10,
        "trey-right": 11,
        "empty": 12,
        "double-tight": 13,
        "heavy": 14,
    }

    def __init__(self, data_source, coco_file, seg_transform=None, classifier_transform=None):
        super().__init__()
        self.data_source=data_source
        self.seg_transform=seg_transform
        self.classifier_transform=classifier_transform

        with open(coco_file, 'r') as f:
            self.coco = json.load(f)

        self.img_folder_path = os.path.join(data_source,"images")
        self.seg_img_folder_path = os.path.join(data_source,"resize_images")
        self.mask_img_folder_path = os.path.join(data_source,"Team_masks","resize_off_masks")
        self.mask_per_player_img_folder_path = os.path.join(data_source,"resize_players_masks")
        self.images = [
            img for img in self.coco['images']
            if img.get('attributes', {}).get('formation', 'unknown') not in self.RARE_FORMATIONS
        ]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_entry = self.images[idx]

        img_path = os.path.join(self.img_folder_path, img_entry['file_name'])
        image = Image.open(img_path).convert("RGB")

        seg_img_path= os.path.join(self.seg_img_folder_path, img_entry['file_name'])
        seg_image = Image.open(seg_img_path).convert("RGB")

        mask_path = os.path.join(self.mask_img_folder_path, img_entry['offense_mask'])
        mask=  Image.open(mask_path).convert("L")



        if self.seg_transform:
            seg_image, mask= self.seg_transform(seg_image, mask)
        if self.classifier_transform:
            image= self.classifier_transform(image)
        else:
            image = transforms.ToTensor()(image)

        meta=img_entry["resize_meta"]
        
        formation_str = img_entry.get('attributes', {}).get('formation', 'unknown')
        formation_label = self.FORMATION_MAP.get(formation_str, -1)
        formation_label = torch.tensor(formation_label, dtype=torch.long)
        
        bboxes = []
        roles = []
        positions = []
        alignments = []
        playerMask = []

        anns = [ann for ann in self.coco.get('annotations', []) if ann['image_id'] == img_entry['id']]
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = next(
                (c["name"] for c in self.coco["categories"] if c["id"] == cat_id),
                None,
            )

            if cat_name not in self.ROLE_MAP:
                continue
            
            role_id = self.ROLE_MAP.get(cat_name, -1)
            roles.append(role_id)

            scale = meta['scale']
            pad_left = meta['pad_left']
            pad_top = meta['pad_top']
            new_w = meta['new_w']
            new_h = meta['new_h']
            
            x, y, w, h = ann['bbox']
            
            x_new = x * scale + pad_left
            y_new = y * scale + pad_top
            w_new = w * scale
            h_new = h * scale
            
            x_norm = x_new / new_w
            y_norm = y_new / new_h
            w_norm = w_new / new_w
            h_norm = h_new / new_h
            
            bboxes.append([x_norm, y_norm, w_norm, h_norm])
            
            position_str = ann.get("position", None)
            position_label = self.POSITION_MAP.get(position_str, -1)
            positions.append(position_label)

            alignment_str = ann.get("alignment", None)
            alignment_label = self.ALIGNMENT_MAP.get(alignment_str, -1)
            alignments.append(alignment_label)

            mask_per_player_path = os.path.join(self.mask_per_player_img_folder_path, ann["segmentation_mask"])
            mask_per_player = Image.open(mask_per_player_path).convert("L")
            playerMask.append(mask_per_player)
        
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        roles = torch.tensor(roles, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.long)
        alignments = torch.tensor(alignments, dtype=torch.long)

        if self.seg_transform:
            playerMask = [self.seg_transform(m)[0] for m in playerMask]
            
        return {
                # "image": image,
                "seg_image": seg_image,
                "mask": mask,
                "meta": meta,
                "formation_label": formation_label,
                "bboxes": bboxes,
                "roles": roles,
                "positions": positions,
                "alignments": alignments,
                "playerMasks": playerMask
            }
        