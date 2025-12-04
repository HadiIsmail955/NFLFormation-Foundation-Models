from torchvision import transforms
from torchvision.transforms import functional as TF

class SAMTransformer:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675/255, 116.28/255, 103.53/255],
                std=[58.395/255, 57.12/255, 57.375/255]
            ),
        ])
        
    def __call__(self, image, mask):
        image_tensor = self.image_transform(image)

        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return image_tensor, mask_tensor
