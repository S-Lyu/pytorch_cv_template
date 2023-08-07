from utils.train_utils import AverageMeter, save_checkpoint, load_checkpoint
from dataset.customdataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from config import Config
from models.net import Net
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

def train(model, train_loader, valid_loader, resume=None):
    train_losses = AverageMeter()
    valid_losses = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    
    if resume is not None:
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, resume
        )
    else:
        best_val_loss = 1
        start_epoch = 0

    for epoch in trange(start_epoch, config.epochs):
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
            
        # Validation
        model.eval()
        pbar = tqdm(valid_loader, leave=False, 
                    desc=f"Valid {epoch+1}/{config.epochs}")
        with torch.no_grad():
            for img, label in pbar:
                output = model(img)
                loss = criterion(output, label)
                valid_losses.update(loss.item())
                pbar.set_postfix_str(f"Loss: {valid_losses.avg:.4f}")
                
        if valid_losses.avg < best_val_loss:
            best_val_loss = valid_losses.avg
            best_epoch = epoch
            checkpoint_path = f'{config.weights_save_path}/model_best.pth'
            save_checkpoint(model, epoch, optimizer, best_val_loss, checkpoint_path)
            print(f"Saved best model @ {epoch+1} with Loss: {valid_losses.avg:.4f}")
            
    print(f"Finished training, best @ {best_epoch+1} with Loss: {best_val_loss:.4f}")
    return        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, type=str) # path of checkpoint
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.to(device)
    train_loader, valid_loader = get_loaders()
    if args.resume:
        train(model, train_loader, valid_loader, resume=args.resume)
    else:
        train(model, train_loader, valid_loader)




