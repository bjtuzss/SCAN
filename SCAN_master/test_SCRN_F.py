import torch
import torch.nn.functional as F
from torch.autograd import Variable

import os
from scipy import misc
import imageio
from datetime import datetime
import numpy as np
from skimage import io
import glob
from measures import compute_ave_MAE_of_methods
from utils.data import test_dataset
from model.ResNet_models import SCRN


class Scan_Master():
    def __init__(self):
        self.datas = []
        # >>>>>>> Follows have to be manually configured <<<<<<< #
        self.data_name = 'GGRNet'  # this will be drawn on the bottom center of the figures
        self.data_dir = 'D:/workspaces--pycharm/flask_scrn_2/SCAN_master/result/bestnet-Finally2'  # set the data directory,
        # ground truth and results to-be-evaluated should be in this directory
        # the figures of PR and F-measure curves will be saved in this directory as well
        self.gt_dir = 'D:/workspaces--pycharm/flask_scrn_2/SCAN_master/gt'  # set the ground truth folder name
        self.rs_dirs = ['']
        # 'ECSSD1','ECSSD2','ECSSD3','ECSSD4','ECSSD5','ECSSD6','ECSSD7','ECSSD8','ECSSD9','ECSSD10','ECSSD11','ECSSD12','ECSSD13','ECSSD14','ECSSD15','ECSSD16','ECSSD17']#,'',''] # set the folder names of different methods
        # 'rs1' contains the result of method1
        # 'rs2' contains the result of method 2
        # we suggest to name the folder as the method names because they will be shown in the figures' legend
        self.lineSylClr = ['r-', 'r', 'c-', 'r', 'g', 'c', 'r-', 'c-', 'r', 'g', 'c', 'r-', 'c-', 'r', 'g', 'c', 'c',
                           'g']  # ,'b-','g-','c-'] # curve style, same size with rs_dirs
        self.linewidth = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2]  # ,2,2,2] # line width, same size with rs_dirs
        # >>>>>>> Above have to be manually configured <<<<<<< #

        self.gt_name_list = glob.glob(self.gt_dir + '/' + '*.png')  # get the ground truth file name list
        # get directory list of predicted maps
        self.rs_dir_lists = []
        # for i in range(len(rs_dirs)):
        self.rs_dir_lists.append(self.data_dir + '/')
        print('\n')

    def handle_scan(self):
        model = SCRN()
        state_dict = torch.load('D:/workspaces--pycharm/flask_scrn_2/SCAN_master/pths/Finally2.pth',
                                map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.cpu()
        model.eval()

        data_path = 'D:/workspaces--pycharm/flask_scrn_2/SCAN_master/resources/'
        # valset = ['ECSSD', 'HKUIS', 'PASCAL', 'DUT-OMRON', 'THUR15K', 'DUTS-TEST']
        valset = ['bestnet-Finally2']
        for dataset in valset:
            save_path = 'D:/workspaces--pycharm/flask_scrn_2/SCAN_master/result/' + dataset + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_root = 'D:/workspaces--pycharm/flask_scrn_2/SCAN_master/resources/'
            gt_root = 'SCAN_master/gt/'
            test_loader = test_dataset(image_root, testsize=352)

            with torch.no_grad():
                for i in range(test_loader.size):
                    image, name = test_loader.load_data()
                    # image, gt, name = test_loader.load_data()
                    # gt = np.array(gt).astype('float')
                    # gt = gt / (gt.max() + 1e-8)
                    image = Variable(image).cpu()

                    res, edge = model(image)

                    # res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
                    res = F.interpolate(res, size=(352, 352), mode='bilinear', align_corners=True)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    imageio.imsave(
                        'D:/workspaces--pycharm/flask_scrn_2/SCAN_master/result/bestnet-Finally2/' + name + '.png', res)

    def getMsg(self):
        self.datas = []
        # 1. =======compute the average MAE of methods=========
        print("------1. Compute the average MAE of Methods------")
        aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(self.gt_name_list, self.rs_dir_lists)
        for i in range(0, len(self.rs_dirs)):
            print(
                '>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.4f' % (
                self.rs_dirs[i], gt2rs_mae[i], len(self.gt_name_list), aveMAE[i]))
            data_MAE = str(aveMAE[i])
            self.datas.append(data_MAE)

        # 2. =======compute the Precision, Recall and F-measure of methods=========
        from measures import compute_PRE_REC_FM_of_methods, plot_save_pr_curves, plot_save_fm_curves
        print("------2. Compute the Precision, Recall and F-measure of Methods------")
        PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(self.gt_name_list, self.rs_dir_lists, beta=0.3)
        for i in range(0, FM.shape[0]):
            print(">>", self.rs_dirs[i], ":", "num_rs/num_gt-> %d/%d," % (int(gt2rs_fm[i][0]), len(self.gt_name_list)),
                  "maxF->%.4f, " % (np.max(FM, 1)[i]), "meanF->%.4f, " % (np.mean(FM, 1)[i]),
                  "PRE->%.4f, " % (np.mean(PRE, 1)[i]), "REC->%.4f, " % (np.mean(REC, 1)[i]))
            data_maxF = np.max(FM, 1)[i]
            data_meanF = np.mean(FM, 1)[i]
            data_PRE = np.mean(PRE, 1)[i]
            data_REC = np.mean(REC, 1)[i]
            self.datas.append(data_maxF)
            self.datas.append(data_meanF)
            self.datas.append(data_PRE)
            self.datas.append(data_REC)
        print('\n')
        print(self.datas)

        # 3. =======Plot and save precision-recall curves=========
        print("------ 3. Plot and save precision-recall curves------")
        plot_save_pr_curves(PRE,  # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                            REC,  # numpy array (num_rs_dir,255)
                            method_names=self.rs_dirs,
                            # method names, shape (num_rs_dir), will be included in the figure legend
                            lineSylClr=self.lineSylClr,  # curve styles, shape (num_rs_dir)
                            linewidth=self.linewidth,  # curve width, shape (num_rs_dir)
                            xrange=(0.5, 1.0),  # the showing range of x-axis
                            yrange=(0.5, 1.0),  # the showing range of y-axis
                            dataset_name=self.data_name,  # dataset name will be drawn on the bottom center position
                            save_dir=self.data_dir,  # figure save directory
                            save_fmt='png')  # format of the to-be-saved figure
        print('\n')

        # 4. =======Plot and save F-measure curves=========
        print("------ 4. Plot and save F-measure curves------")
        plot_save_fm_curves(FM,  # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                            mybins=np.arange(0, 256),
                            method_names=self.rs_dirs,
                            # method names, shape (num_rs_dir), will be included in the figure legend
                            lineSylClr=self.lineSylClr,  # curve styles, shape (num_rs_dir)
                            linewidth=self.linewidth,  # curve width, shape (num_rs_dir)
                            xrange=(0.0, 1.0),  # the showing range of x-axis
                            yrange=(0.0, 1.0),  # the showing range of y-axis
                            dataset_name=self.data_name,  # dataset name will be drawn on the bottom center position
                            save_dir=self.data_dir,  # figure save directory
                            save_fmt='png')  # format of the to-be-saved figure
        print(self.datas)
        return self.datas
