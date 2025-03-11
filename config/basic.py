import torchvision.transforms as transforms
from PIL import Image

class ConfigBasic:
    def __init__(self,):
        self.dataset = None
        self.setting = None
        self.logscale = False
        self.set_optimizer_parameters()
        self.set_training_opts()

    def set_dataset(self):
        if self.dataset == 'SICE':
            self.is_filelist = True
            self.train_file = './Dataset_txt/SICEV2_Train.txt'
            self.test_file = './Dataset_txt/SICEV2_Test.txt'
            self.delimeter = ","
            self.img_idx = 3
            self.gt_idx = 4
            self.lb_idx = 1

        elif self.dataset == 'MSEC':
            self.is_filelist = True
            self.train_file = './Dataset_txt/MSEC_Train.txt'
            self.test_file = './Dataset_txt/MSEC_Test.txt'
            self.delimeter = ","
            self.img_idx = 3
            self.gt_idx = 4
            self.lb_idx = 1
        
        elif self.dataset == 'LCDP':
            self.is_filelist = True
            self.train_file = './Dataset_txt/LCDP_Train.txt'
            self.test_file = './Dataset_txt/LCDP_Test.txt'
            self.delimeter = ","
            self.img_idx = 3
            self.gt_idx = 4
            self.lb_idx = 1

        else:
            raise ValueError(f'{self.dataset} is out of range!')

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform = transforms.Compose([lambda x: Image.fromarray(x), transforms.ToTensor(),])


    def set_optimizer_parameters(self):
        # *** Optimizer
        self.adam = True
        self.learning_rate = 0.0001
        self.lr_decay_epochs = [30, 50, 100]
        self.lr_decay_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 0.0005

        # *** Scheduler
        self.scheduler = 'cosine'

    def set_training_opts(self):
        # *** Print Option
        self.val_freq = 3
        self.print_freq = 50

        # *** Training
        self.batch_size = 4
        self.num_workers = 2
        self.epochs = 100

        # *** Save option
        self.save_freq = 100
        self.wandb = False
