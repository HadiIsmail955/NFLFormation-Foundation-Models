import os
import json
import torch
import random
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
        "ace-left": 1,
        "ace-right": 2,
        "trips-left": 3,
        "trips-right": 4,
        "twins-right": 5,
        "bunch-left": 6,
        "bunch-right": 7,
        "i-formation": 8,
        "trey-left": 9,
        "trey-right": 10,
        "empty": 11,
        "double-tight": 12,
        "heavy": 13,
    }

    FLIP_MAP = {
        "trips-left": "trips-right",
        "trips-right": "trips-left",
        "bunch-left": "bunch-right",
        "bunch-right": "bunch-left",
        "trey-left": "trey-right",
        "trey-right": "trey-left",
        "ace-left": "ace-right",
        "ace-right": "ace-left",
    }

    def __init__(self, data_source, coco_file, seg_transform=None, classifier_transform=None, enable_flip=False, flip_prob=0.5):
        super().__init__()
        self.data_source=data_source
        self.seg_transform=seg_transform
        self.classifier_transform=classifier_transform
        self.enable_flip=enable_flip
        self.flip_prob=flip_prob
        with open(coco_file, 'r') as f:
            self.coco = json.load(f)
        from collections import Counter

        labels = [img["attributes"]["formation"] for img in self.coco["images"]
                if img.get("attributes", {}).get("formation") in self.FORMATION_MAP]
        
        print(Counter(labels))

        self.img_folder_path = os.path.join(data_source,"images")
        self.seg_img_folder_path = os.path.join(data_source,"resize_images")
        self.mask_img_folder_path = os.path.join(data_source,"Team_masks","resize_off_masks")
        self.mask_per_player_img_folder_path = os.path.join(data_source,"resize_players_masks")
        self.images = [
            img for img in self.coco['images']
            if img.get('attributes', {}).get('formation') in self.FORMATION_MAP
        ]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        do_flip = self.enable_flip and random.random() < self.flip_prob

        img_entry = self.images[idx]

        img_path = os.path.join(self.img_folder_path, img_entry['file_name'])
        image = Image.open(img_path).convert("RGB")

        seg_img_path= os.path.join(self.seg_img_folder_path, img_entry['file_name'])
        seg_image = Image.open(seg_img_path).convert("RGB")

        mask_path = os.path.join(self.mask_img_folder_path, img_entry['offense_mask'])
        mask=  Image.open(mask_path).convert("L")



        if self.seg_transform:
            seg_image, mask = self.seg_transform.apply(
                image=seg_image,
                mask=mask,
                do_flip=do_flip,
            )
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
        playerMasks = []

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

            if do_flip:
                x_norm = 1.0 - (x_norm + w_norm)
            
            bboxes.append([x_norm, y_norm, w_norm, h_norm])
            
            position_str = ann.get("position", None)
            position_label = self.POSITION_MAP.get(position_str, -1)
            positions.append(position_label)

            alignment_str = ann.get("alignment", None)
            alignment_label = self.ALIGNMENT_MAP.get(alignment_str, -1)
            alignments.append(alignment_label)

            mask_per_player_path = os.path.join(self.mask_per_player_img_folder_path, ann["segmentation_mask"])
            mask_per_player = Image.open(mask_per_player_path).convert("L")
            _, pm = self.seg_transform.apply(
                image=None,
                mask=mask_per_player,
                do_flip=do_flip,
            )
            playerMasks.append(pm)
        
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        roles = torch.tensor(roles, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.long)
        alignments = torch.tensor(alignments, dtype=torch.long)
            
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
                "playerMasks": playerMasks
            }
        