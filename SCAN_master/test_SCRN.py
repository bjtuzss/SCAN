import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os
from scipy import misc
import imageio
from datetime import datetime
from PIL import Image
from utils.data import test_dataset
from model.ResNet_models import SCRN


def handle_scan():
    model = SCRN()
    state_dict = torch.load('SCAN_master/Finally2.pth', map_location=torch.device('cpu'))
    # create new OrderedDict that does not contain `module.`
    model.load_state_dict(state_dict)
    model.cpu()
    model.eval()

    data_path = './'
    # valset = ['ECSSD', 'HKUIS', 'PASCAL', 'DUT-OMRON', 'THUR15K', 'DUTS-TEST']
    valset = ['bestnet-Finally2']
    for dataset in valset:
        save_path = 'SCAN_master/result/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = 'SCAN_master/resources/'
        # gt_root = data_path + '/gt/'
        test_loader = test_dataset(image_root, testsize=352)

        with torch.no_grad():
            for i in range(test_loader.size):
                image, name = test_loader.load_data()
                # gt = np.array(gt).astype('float')
                # gt = gt / (gt.max() + 1e-8)
                image = Variable(image).cpu()

                res, edge = model(image)

                res = F.interpolate(res, size=(352, 352), mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze() * 255
                res1 = res
                res = Image.fromarray(res.astype(np.uint8))
                res = res.convert('RGB')
                # print(res)
                rows = res.size[0]
                cols = res.size[1]
                # print(image)
                image = image.data.squeeze().cpu().numpy()
                # image=np.transpose(image, [1, 2, 0]) * 255
                image = (image[0] + 1) / 2 * 255
                image = Image.fromarray(image.astype(np.uint8))
                image = image.convert('RGB')
                for i in range(0, rows):
                    for j in range(0, cols):
                        img2 = (res.getpixel((i, j)))
                        # print(img2)
                        if (img2[0] > 50 or img2[1] > 50 or img2[2] > 50):
                            image.putpixel((i, j), (234, 53, 57, 255))
                        # image.putpixel((i, j), (234, 53, 57, 255))
                image = image.convert('RGB')
                imageio.imsave(save_path + name + '.png', image)
                imageio.imsave(save_path + name + 'binary.png', res1)
                # imageio.imsave(save_path + name + '.png', res)
