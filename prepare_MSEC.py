import glob

MSEC_PATH = '/home/daehyun/Lowlight_models/data/MSEC' # Add your MSEC dataset path in here

prepare_type = 'test' # train, test


if prepare_type == 'train':
    ## Training set
    GT_Path = MSEC_PATH + '/training/GT_IMAGES'
    TrainsetGT = glob.glob(GT_Path + '/*.jpg')

    f = open("Dataset_txt/MSEC_Train.txt", 'w')

    f.write(",Ex,filename,filepath,gtpath,MSEC\n")
    # _0, _N1.5, _N1, _P1, _P1.5
    # Low exposure datasets
    num = 0
    for gt_path in TrainsetGT:
        input_path = gt_path.replace('GT_IMAGES', 'INPUT_IMAGES')

        path_name = input_path.split('/')[-1][:-4]
        ## 0
        Ex = '0'
        path_name_0 = path_name + '_0'
        input_path_0 = input_path.replace(path_name+'.jpg', path_name_0+'.JPG')
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1.5
        # Ex = '-2'
        Ex = '-1'
        path_name_0 = path_name + '_N1.5'
        input_path_0 = input_path.replace(path_name+'.jpg', path_name_0+'.JPG')
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1
        Ex = '-1'
        path_name_0 = path_name + '_N1'
        input_path_0 = input_path.replace(path_name+'.jpg', path_name_0+'.JPG')
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## P1
        Ex = '1'
        path_name_0 = path_name + '_P1'
        input_path_0 = input_path.replace(path_name+'.jpg', path_name_0+'.JPG')
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## P1.5
        # Ex = '2'
        Ex = '1'
        path_name_0 = path_name + '_P1.5'
        input_path_0 = input_path.replace(path_name+'.jpg', path_name_0+'.JPG')
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)

    f.close()


elif prepare_type == 'test':
    ## Test set
    GT_Path = MSEC_PATH + '/testing/expert_c_testing_set'
    INPUT_path = MSEC_PATH + '/testing/INPUT_IMAGES'
    TestsetGT = glob.glob(GT_Path + '/*.jpg')

    f = open("Dataset_txt/MSEC_Test.txt", 'w')

    f.write(",Ex,filename,filepath,gtpath,MSEC\n")
    # _0, _N1.5, _N1, _P1, _P1.5
    # Low exposure datasets
    num = 0
    for gt_path in TestsetGT:

        path_name = gt_path.split('/')[-1][:-4]
        ## 0
        Ex = '0'
        path_name_0 = path_name + '_0'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1.5
        Ex = '-2'
        path_name_0 = path_name + '_N1.5'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1
        Ex = '-1'
        path_name_0 = path_name + '_N1'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## P1
        Ex = '1'
        path_name_0 = path_name + '_P1'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1.5
        Ex = '2'
        path_name_0 = path_name + '_P1.5'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)

    f.close()


elif prepare_type == 'test_val':
    ## Test set
    GT_Path = MSEC_PATH + '/validation/GT_IMAGES'
    INPUT_path = MSEC_PATH + '/validation/INPUT_IMAGES'
    TestsetGT = glob.glob(GT_Path + '/*.jpg')

    f = open("Dataset_txt/MSEC_Test.txt", 'w')

    f.write(",Ex,filename,filepath,gtpath,MSEC\n")
    # _0, _N1.5, _N1, _P1, _P1.5
    # Low exposure datasets
    num = 0
    for gt_path in TestsetGT:

        path_name = gt_path.split('/')[-1][:-4]
        ## 0
        Ex = '0'
        path_name_0 = path_name + '_0'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1.5
        Ex = '-2'
        path_name_0 = path_name + '_N1.5'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1
        Ex = '-1'
        path_name_0 = path_name + '_N1'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## P1
        Ex = '1'
        path_name_0 = path_name + '_P1'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)
        ## N1.5
        Ex = '2'
        path_name_0 = path_name + '_P1.5'
        input_path_0 = INPUT_path + '/' + path_name_0 + '.JPG'
        Num = str(num)
        num += 1
        data = Num + ',' + Ex + ',' + path_name_0 + ',' + input_path_0 + ',' + gt_path + ',' + 'MSEC\n'
        f.write(data)

    f.close()


else:
    print("Wrong dataset type.")

