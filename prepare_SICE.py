import glob

SICEV2_PATH = '/home/daehyun/Lowlight_models/data/SICEV2' # add your SICE dataset path in here

prepare_type = 'train' # train, test


if prepare_type == 'train':
    ## Training set
    GT_Path = SICEV2_PATH + '/train/gt' # gt, low, over
    path = SICEV2_PATH + '/train'
    TrainsetGT = glob.glob(GT_Path + '/*.JPG')

    f = open("Dataset_txt/SICEV2_Train.txt", 'w')
    f.write(",Ex,filename,filepath,gtpath,SICEV2\n")

    coin = 0

    # Low exposure datasets
    for gt_path in TrainsetGT:
        low_path = gt_path.replace('gt', 'low')
        over_path = gt_path.replace('gt', 'over')

        # low
        path_name_low = low_path.split('/')[-1]
        Ex = '-1'
        Num = low_path.split('/')[-1].split('.')[0]
        data = Num + ',' + Ex + ',' + path_name_low + ',' + low_path + ',' + gt_path + ',' + 'SICEV2\n'
        f.write(data)
        # over
        path_name_over = over_path.split('/')[-1]
        Ex = '1'
        Num = over_path.split('/')[-1].split('.')[0]
        data = Num + ',' + Ex + ',' + path_name_over + ',' + over_path + ',' + gt_path + ',' + 'SICEV2\n'
        f.write(data)
    f.close()


elif prepare_type == 'test':
    ## Test set
    GT_Path = SICEV2_PATH + '/test/gt' # gt, low, over
    path = SICEV2_PATH + '/test'
    TestsetGT = glob.glob(GT_Path + '/*.JPG')

    f = open("Dataset_txt/SICEV2_Test.txt", 'w')
    f.write(",Ex,filename,filepath,gtpath,SICEV2\n")

    # Low exposure datasets
    for gt_path in TestsetGT:
        low_path = gt_path.replace('gt', 'low')
        over_path = gt_path.replace('gt', 'over')

        # low
        path_name = low_path.split('/')[-1]
        Ex = '0'
        Num = low_path.split('/')[-1].split('.')[0]
        data = Num + ',' + Ex + ',' + path_name + ',' + low_path + ',' + gt_path + ',' + 'SICEV2\n'
        f.write(data)
        # over
        path_name = over_path.split('/')[-1]
        Ex = '2'
        Num = over_path.split('/')[-1].split('.')[0]
        data = Num + ',' + Ex + ',' + path_name + ',' + over_path + ',' + gt_path + ',' + 'SICEV2\n'
        f.write(data)
    f.close()

else:
    print("Wrong dataset type.")

