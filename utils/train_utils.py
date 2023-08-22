import torch

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path):
    state = {
       'epoch': epoch + 1,
       'state_dict': model.state_dict(),
       'optimizer': optimizer.state_dict(),
       'best_val_loss': best_val_loss
    }
    torch.save(state, checkpoint_path)
    return

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    return model, optimizer, start_epoch, best_val_loss
