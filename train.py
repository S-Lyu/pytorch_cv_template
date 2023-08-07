from dataset.customdataset import CustomDataset
from utils.train_utils import AverageMeter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from config import Config
from models.net import Net
from glob import glob
import tensorboard
import argparse
import torch


config = Config()

def get_loaders():
    trainset = CustomDataset(dataset_path=config.trainset_path,
                             anno_path=config.train_anno_path,
                             resize=config.resize)
    validset = CustomDataset(dataset_path=config.validset_path,
                             anno_path=config.valid_anno_path,
                             resize=config.resize)
    train_loader = DataLoader(trainset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              pin_memory=config.pin_memory,
                              shuffle=True)
    valid_loader = DataLoader(validset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              pin_memory=config.pin_memory)
    return train_loader, valid_loader

def train(model, train_loader, valid_loader):
    train_losses = AverageMeter()
    valid_losses = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    
    best = 1
    for epoch in trange(config.epochs):
        # training
        model.train()
        pbar = tqdm(train_loader, leave=False, 
                    desc=f"Train {epoch+1}/{config.epochs}")
        for img, label in pbar:
            output = model(img)
            loss = criterion(output, label)
            train_losses.update(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"Loss: {train_losses.avg:.4f}")
            
        # Validate
        model.eval()
        pbar = tqdm(valid_loader, leave=False, 
                    desc=f"Valid {epoch+1}/{config.epochs}")
        with torch.no_grad():
            for img, label in pbar:
                output = model(img)
                loss = criterion(output, label)
                valid_losses.update(loss.item())
                pbar.set_postfix_str(f"Loss: {valid_losses.avg:.4f}")
                
        if valid_losses.avg < best:
            best = valid_losses.avg
            best_epoch = epoch
            torch.save(f'{config.weights_save_path}/model_best.pth')
            print(f"Saved best model @ {epoch+1} with Loss: {valid_losses.avg:.4f}")
            
    print(f"Finished training, best @ {best_epoch+1} with Loss: {best:.4f}")
    return        
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.to(device)
    train_loader, valid_loader = get_loaders()
    train(model, train_loader, valid_loader)



