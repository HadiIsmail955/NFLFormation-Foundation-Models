import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class PresnapDataset(Dataset):
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

        self.images = self.coco['images']

        self.formation_map = {
            "shotgun": 0,
            "singleback": 1,
            "pistol": 2,
            "ace": 3
        }

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_entry = self.images[idx]

        img_path = os.path.join(self.img_folder_path, img_entry['file_name'])
        image = Image.open(img_path).convert("RGB")

        seg_img_path= os.path.join(self.seg_img_folder_path, img_entry['file_name'])
        seg_image = Image.open(seg_img_path).convert("RGB")

        mask_path = os.path.join(self.mask_img_folder_path, img_entry['offense_mask'])
        mask= mask = Image.open(mask_path).convert("L")

        if self.seg_transform:
            seg_image, mask= self.seg_transform(seg_image, mask)
        if self.classifier_transform:
            image= self.classifier_transform(image)

        meta=img_entry["resize_meta"]
        formation_label=None
        if image["formation"]:
            formation_str = img_entry.get('formation', 'unknown')
            formation_label = self.formation_map.get(formation_str, -1)  
            formation_label = torch.tensor(formation_label, dtype=torch.long)

        return image, seg_image, mask, meta, formation_label

        



        
        