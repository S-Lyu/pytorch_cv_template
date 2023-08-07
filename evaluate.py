from dataset.customdataset import CustomDataset
from utils.train_utils import AverageMeter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from config import Config
from models.net import Net
from glob import glob
import numpy as np 
import tensorboard
import torch


config = Config()

def evaluate(model, test_loader):
    test_losses = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    pbar = tqdm(test_loader, leave=False, desc=f"Test")
    with torch.no_grad():
        for img, label in pbar:
            output = model(img)
            loss = criterion(output, label)
            test_losses.update(loss.item())
            pbar.set_postfix_str(f"Loss: {test_losses.avg:.4f}")
            
    print(f"Finished testing with Loss: {test_losses.avg:.4f}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.to(device)
    testset = CustomDataset(dataset_path=config.testset_path,
                            anno_path=config.test_anno_path,
                            resize=config.resize)
    test_loader = DataLoader(testset,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             pin_memory=config.pin_memory)
    evaluate(model, test_loader)
    







