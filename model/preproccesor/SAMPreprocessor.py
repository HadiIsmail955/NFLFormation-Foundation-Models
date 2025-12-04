import torch
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
class SAMPreprocessor:
    def __init__(self, target_size=1024):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1,3,1,1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1,3,1,1)

    def __call__(self, image):
        """
        image: numpy array (H,W,3) in RGB
        returns:
            tensor: (1,3,1024,1024)
            meta: dictionary needed for reversing mask
        """
        resized = self.transform.apply_image(image)
        h, w = resized.shape[:2]

        x = torch.tensor(resized).permute(2,0,1).float()

        x = (x - self.pixel_mean[0]) / self.pixel_std[0]

        x = x.unsqueeze(0)  

        return x
