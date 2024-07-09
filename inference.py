from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from datasets.dataset import build_data_loader
from models.unet import UNet as u_net
from models.hrnet import HRNetV2
from datasets.metric import *
from models.md_net import Multi_decoder_Net, Two_decoder_Net
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, confusion_matrix

print("PyTorch Version: ", torch.__version__)

'''
inference
'''


def infer_model(opts):
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']
    load_epoch = opts['load_epoch']
    gpus = opts["gpu_list"].split(',')
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

    eval_data_dir = opts["eval_data_dir"]

    train_dir = opts["train_dir"]
    model_type = opts['model_type']
    fractal_dir = opts['fractal_dir']
    dataset_name = opts["dataset_name"]

    model_score_dir = os.path.join(str(os.path.split(train_dir)[0]),
                                   'predict_score/' + dataset_name + '_' + str(load_epoch))
    if not os.path.exists(model_score_dir): os.makedirs(model_score_dir)

    # dataloader
    print("==> Create dataloader")
    dataloader = build_data_loader(dataset_name, eval_data_dir, val_batch_size, dataset_type, is_train=False,
                                   fractal_dir=fractal_dir)

    # define network
    print("==> Create network")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if 'fractal' in opts["dataset_type"]:
        if 'road' in opts["dataset_type"]:
            if 'RGB' in fractal_dir:
                num_channels = 6
            else:
                num_channels = 5
        else:
            num_channels = 3
    else:
        if 'road' in opts["dataset_type"] or 'copy' in opts["dataset_type"]:
            num_channels = 3
        else:
            num_channels = 1

    num_classes = 1
    if model_type == 'unet':
        model = u_net(num_channels, num_classes)
    elif model_type == 'hrnet':
        model = HRNetV2(n_channels=num_channels, n_class=num_classes)
    elif model_type == 'Two_decoder_Net':
        model = Two_decoder_Net(num_channels, num_classes)
    elif model_type == 'Multi_decoder_Net':
        model = Multi_decoder_Net(num_channels, num_classes)


    # load trained model
    pretrain_model = os.path.join(train_dir, str(load_epoch) + ".pth")
    # print(pretrain_model)
    # pretrain_model = os.path.join(train_dir, "checkpoints_" + str(load_epoch) + ".pth")

    if os.path.isfile(pretrain_model):
        c_checkpoint = torch.load(pretrain_model)
        model.load_state_dict(c_checkpoint["model_state_dict"])
        print("==> Loaded pretrianed model checkpoint '{}'.".format(pretrain_model))
    else:
        print("==> No trained model.")
        return 0

    # set model to gpu mode
    print("==> Set to GPU mode")

    model.cuda()
    model = torch.nn.DataParallel(model)

    # enable evaluation mode
    with torch.no_grad():
        model.eval()
        total_img = 0
        for inputs in dataloader:
            images = inputs["image"].cuda()
            img_name = inputs['ID']
            # print('now process image is %s' % (img_name))
            total_img += len(images)
            # unet
            if model_type == 'unet':
                p_seg = model(images)
            elif model_type == 'hrnet':
                outputs_list = model(images)
                p_seg = outputs_list[0]
            elif model_type == 'Two_decoder_Net':
                p_seg, pred_bone = model(images)
            elif model_type == 'Multi_decoder_Net':
                p_seg, pred_bone, pred_edge = model(images)


            for i in range(len(images)):
                # print('predict image: {}'.format(img_name[i]))
                    now_dir = model_score_dir
                    os.makedirs(now_dir, exist_ok=True)
                    np.save(os.path.join(now_dir, img_name[i].split('.')[0] + '.npy'),
                            p_seg[i][0].cpu().numpy().astype(np.float32))
                    cv2.imwrite(os.path.join(now_dir, img_name[i].split('.')[0] + '.tif'),
                                p_seg[i][0].cpu().numpy().astype(np.float32))




if __name__ == "__main__":
    model_choice = ['unet', 'hrnet', 'Two_decoder_Net', 'Multi_decoder_Net']
    dataset_list = ['er', 'er_fractal', 'er_fractal_two_decoder', 'nucleus_fractal_two_decoder','nucleus_fractal_two_decoder_weighted']
    txt_choice = ['train_mito.txt', 'test_mito.txt', 'train_er.txt', 'test_er.txt', 'test_stare.txt', 'train_stare.txt']

    opts = dict()
    opts["dataset_name"] = "ER"
    opts['dataset_type'] = 'er_fractal'
    opts["eval_batch_size"] = 1
    opts["gpu_list"] = "0,1,2,3"
    opts["train_dir"] = "./train_logs/er_fractal_HRNet_iou_32_0.05_50_0.3_1000_20240312_warmup/checkpoints"
    opts["eval_data_dir"] = "./dataset_txts/test_er.txt"
    opts["decoder_type"] = "edge"
    opts['model_type'] = 'hrnet'
    opts["load_epoch"] = 'best'

    opts["fractal_dir"] = 'Fractal_info_5'

    best_iou = 0.0
    infer_model(opts)

