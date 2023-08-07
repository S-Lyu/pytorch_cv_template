try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch


class CustomDataset(Dataset):
    
    def __init__(self, resize=[]):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __getitem__(self, index):
    # transform(imgs)  # Image Classification
    # transform(videos)  # Video Tasks
    # transform(imgs, bboxes, labels)  # Object Detection
    # transform(imgs, bboxes, masks, labels)  # Instance Segmentation
    # transform(imgs, masks)  # Semantic Segmentation
    # transform({"image": imgs, "box": bboxes, "tag": labels})  # Arbitrary Structure
        return
        
    def __len__(self):
        return 