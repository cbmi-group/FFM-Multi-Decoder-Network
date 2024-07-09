import os
import torch
from torch.utils.data import Dataset
from data_preprocess.pre_process import *


def img_PreProc_nucleus(img, pro_type):
    if pro_type == "clahe":
        img = img_clahe(img)
        img = img / 255.
        sd_img = standardization(img)
        return sd_img

    elif pro_type == "clahe_new":
        img = img_clahe(img)
        return img / 255.


    elif pro_type == "invert":
        img = 255 - img
        return img / 255.

    elif pro_type == "edgeEnhance":
        edge = sober_filter(img)
        edge = edge / np.max(edge)
        return ((img / 255.) + edge) * 0.5

    elif pro_type == "norm_single":
        img = img / 255.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img
    elif pro_type == "norm_dataset":
        img = (img - (1, 1, 1)) / (1, 1, 1)
        return img
    elif pro_type == "clahe_norm_single":
        img = img_clahe(img)
        img = img / 255.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img
    elif pro_type == "clahe_norm":
        img = img_clahe(img)
        img = img / 255.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img


class NUCLEUS_Dataset(Dataset):
    def __init__(self, txt, dataset_type, train, fractal_dir='', weight_dir='', edge_dir='',skeleton_dir='',decoder_type='', log_file='', epoch=0, update_d=5,
                 img_size=256):
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.train = train
        self.fractal_dir = fractal_dir
        self.weight_dir = weight_dir
        self.decoder_type = decoder_type
        self.log_file = log_file
        self.epoch = epoch
        self.update_d = update_d
        self.edge_dir =edge_dir
        self.skeleton_dir = skeleton_dir

        with open(txt, "r") as fid:
            lines = fid.readlines()

        img_mask_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            img_mask_paths.append((words[0], words[1]))

        self.img_mask_paths = img_mask_paths

    def __getitem__(self, index):
        img_path, mask_path = self.img_mask_paths[index]

        # initialize input
        if self.dataset_type == 'nucleus':
            img = cv2.imread(img_path, 0)
            mask = cv2.imread(mask_path, 0)
            img_ = img_PreProc_nucleus(img, pro_type='clahe_norm')

            image_chw = torch.from_numpy(img_).unsqueeze(dim=0).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            sample = {"image": image_chw,
                      "mask": mask_,
                      "ID": os.path.split(img_path)[1]}
        elif self.dataset_type == 'nucleus_fractal':
            npy_path = img_path.replace("images", self.fractal_dir)
            npy_path = npy_path[:-4] + ".npy"
            fractal_info = np.load(npy_path)
            img_FD, img_FL = npy_PreProc(fractal_info)
            img = cv2.imread(img_path, 0)
            mask = cv2.imread(mask_path, 0)
            img_ = img_PreProc_nucleus(img, pro_type='clahe_norm')

            fractal_img = np.stack((img_, img_FD, img_FL), axis=2)
            image_chw = np.transpose(fractal_img, (2, 0, 1))
            image_chw = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            sample = {"image": image_chw,
                      "mask": mask_,
                      "ID": os.path.split(img_path)[1]}
        elif self.dataset_type == 'nucleus_fractal_two_decoder':
            npy_path = img_path.replace("images", self.fractal_dir)
            npy_path = npy_path[:-4] + ".npy"
            fractal_info = np.load(npy_path)
            img_FD, img_FL = npy_PreProc(fractal_info)
            img = cv2.imread(img_path, 0)
            mask = cv2.imread(mask_path, 0)
            img_ = img_PreProc_nucleus(img, pro_type='clahe_norm')

            fractal_img = np.stack((img_, img_FD, img_FL), axis=2)
            image_chw = np.transpose(fractal_img, (2, 0, 1))
            image_chw = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            if self.train:
                edge_path = mask_path.replace('masks_train_aug', self.edge_dir)
                edge = cv2.imread(edge_path, -1)
                edge_ = torch.from_numpy(edge / 255.).unsqueeze_(dim=0).float()
                sample = {"image": image_chw,
                          "mask": mask_,
                          "edge": edge_,
                          "ID": os.path.split(img_path)[1]}
            else:
                sample = {"image": image_chw,
                          "mask": mask_,
                          "ID": os.path.split(img_path)[1]}

        elif self.dataset_type == 'nucleus_fractal_two_decoder_weighted':
            train_imgpth_list = img_path.split('/')
            image_name = train_imgpth_list[-1]
            train_imgpth_list[-2] = train_imgpth_list[-2] + '_' + self.fractal_dir
            npy_path = '/'.join(train_imgpth_list)
            npy_path = npy_path[:-4] + ".npy"

            fractal_info = np.load(npy_path)
            img_FD, img_FL = npy_PreProc(fractal_info)

            img = cv2.imread(img_path, 0)
            mask = cv2.imread(mask_path, 0)

            weight_path = mask_path.replace('masks_train_aug', 'masks_train_aug' + self.weight_dir)
            weighted_npy = weight_path.replace(".tif", ".npy")
            weight1 = np.ones_like(img)
            weight2 = np.load(weighted_npy)
            weight = weight1 + (weight2 / np.max(weight2))

            img_ = img_PreProc_nucleus(img, pro_type='clahe_norm')

            fractal_img = np.stack((img_, img_FD, img_FL), axis=2)
            image_chw = np.transpose(fractal_img, (2, 0, 1))
            image_chw = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()
            weight_ = torch.from_numpy(weight / 1.0).unsqueeze_(dim=0).float()

            if self.train:
                edge_path = mask_path.replace('masks_train_aug', self.edge_dir)
                edge = cv2.imread(edge_path, -1)
                edge_ = torch.from_numpy(edge / 255.).unsqueeze_(dim=0).float()
                sample = {"image": image_chw,
                          "mask": mask_,
                          "edge": edge_,
                          "weight": weight_,
                          "ID": os.path.split(img_path)[1]}
            else:
                sample = {"image": image_chw,
                          "mask": mask_,
                          "ID": os.path.split(img_path)[1]}

        return sample

    def __len__(self):
        return len(self.img_mask_paths)
