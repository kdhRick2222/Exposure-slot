import glob
from PIL import Image
import numpy as np

LCDP_PATH = '/home/daehyun/Lowlight_models/data/LCDP' # Add your LCDP dataset path in here

prepare_type = 'train' # train, test


if prepare_type == 'train':
    ## Training set
    GT_Path = LCDP_PATH + '/train-gt' # gt, low, over
    path = LCDP_PATH + '/train-input'
    TrainsetGT = glob.glob(GT_Path + '/*.png')

    f = open("Dataset_txt/LCDP_Train.txt", 'w')
    f.write(",Ex,filename,filepath,gtpath,LCDP\n")

    # Low exposure datasets
    for gt_path in TrainsetGT:
        low_path = gt_path.replace('train-gt', 'train-input')

        gt_np = np.array(Image.open(gt_path).convert('RGB'))/255.
        low_np = np.array(Image.open(low_path).convert('RGB'))/255.
        diff_np = low_np - gt_np

        # # low
        path_name = low_path.split('/')[-1]
        if diff_np.mean() < 0:
            Ex = '-1'
        elif diff_np.mean() > 0:
            Ex = '1'
        else:
            Ex = '0'
        Num = low_path.split('/')[-1].split('.')[0]
        data = Num + ',' + Ex + ',' + path_name + ',' + low_path + ',' + gt_path + ',' + 'LCDP\n'
        f.write(data)

    f.close()


elif prepare_type == 'test':
    ## Test set
    GT_Path = LCDP_PATH + '/test-gt' # gt, low, over
    path = LCDP_PATH + '/test-input'
    TestsetGT = glob.glob(GT_Path + '/*.png')

    f = open("Dataset_txt/LCDP_Test.txt", 'w')
    f.write(",Ex,filename,filepath,gtpath,LCDP\n")

    # Low exposure datasets
    for gt_path in TestsetGT:
        low_path = gt_path.replace('test-gt', 'test-input')

        gt_np = np.array(Image.open(gt_path).convert('RGB'))/255.
        low_np = np.array(Image.open(low_path).convert('RGB'))/255.
        diff_np = low_np - gt_np
        
        # low
        path_name = low_path.split('/')[-1]
        if diff_np.mean() < 0:
            Ex = '-1'
        elif diff_np.mean() > 0:
            Ex = '1'
        else:
            Ex = '0'
        Num = low_path.split('/')[-1].split('.')[0]
        data = Num + ',' + Ex + ',' + path_name + ',' + low_path + ',' + gt_path + ',' + 'LCDP\n'
        f.write(data)

    f.close()


else:
    print("Wrong dataset type.")

