import torch


class Config:
    
    def __init__(self):
        self.trainset_path = 'dataset/train'
        self.validset_path = 'dataset/valid'
        self.testset_path = 'dataset/test'
        
        self.batch_size = 16
        self.lr = 1e-4
        self.epochs = 30
        self.weights_path = ''
        self.use_multigpus = False
        
    