import torch


class Config:
    
    def __init__(self):
        self.trainset_path = 'dataset/train'
        self.validset_path = 'dataset/valid'
        self.testset_path = 'dataset/test'
        self.train_anno_path = 'dataset/train/[ANNO_PATH]'
        self.valid_anno_path = 'dataset/valid/[ANNO_PATH]'
        self.test_anno_path = 'dataset/test/[ANNO_PATH]'
        self.weights_save_path = 'models/weights'
        self.pretrained_weights = 'models/weights/[WEIGHTS_PATH]'
        self.resize = [512, 512]

        self.lr = 1e-4
        self.weight_decay = 1e-2
        self.epochs = 30        
        self.batch_size = 16
        self.num_workers = 6
        self.pin_memory = True
        self.weights_path = ''
        self.use_multigpus = False
        
    