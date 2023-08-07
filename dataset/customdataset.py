try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    raise ImportError("torchvision.__version__ >= 0.15 is required")
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import torch
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, dataset_path, anno_path, resize=[]):
        super().__init__()
        self.dataset_path = dataset_path
        self.anno_path = anno_path
        self.imgs = sorted(glob(dataset_path + '/.jpg')) # Change it!
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __getitem__(self, index):
        # This requires torchvision.__version__ >= 0.15
        # transform(imgs)  # Image Classification
        # transform(videos)  # Video Tasks
        # transform(imgs, bboxes, labels)  # Object Detection
        # transform(imgs, bboxes, masks, labels)  # Instance Segmentation
        # transform(imgs, masks)  # Semantic Segmentation
        # transform({"image": imgs, "box": bboxes, "tag": labels})  # Arbitrary Structure
        img = Image.open(self.imgs[index]).convert('RGB')
        bbox = self.anno['bbox'][index] # Change it!
        img, bbox = self.transform(img, bbox)
        return img.to(device), bbox.to(device)
        
    def __len__(self):
        return len(self.imgs)