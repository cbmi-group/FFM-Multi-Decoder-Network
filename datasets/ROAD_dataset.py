import os
import torch
from torch.utils.data import Dataset, DataLoader
from data_preprocess.pre_process import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def img_PreProc_road(img, pro_type):
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
        img = (img - (109.244987, 110.007784, 100.735999)) / (74.424917, 72.786659, 75.802716)
        # img = (img - (109.139851, 109.901530, 100.629362)) / (73.187326, 71.655098, 75.030130)
        # img = (img - (109.135833, 109.898700, 100.626847)) / (73.189280, 71.656865, 75.031641)
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


def random_crop(images, labels, aim_size):
    trans = transforms.Compose([transforms.RandomCrop(aim_size)])
    seed = torch.random.seed()
    torch.random.manual_seed(seed)
    cropped_img = trans(images)
    torch.random.manual_seed(seed)
    cropped_label = trans(labels)
    return cropped_img, cropped_label


def random_crop_with_edge_skeleton_weight(images, labels, edge, skeleton, weight, aim_size):
    trans = transforms.Compose([transforms.RandomCrop(aim_size)])
    seed = torch.random.seed()

    torch.random.manual_seed(seed)
    cropped_img = trans(images)
    torch.random.manual_seed(seed)
    cropped_label = trans(labels)
    torch.random.manual_seed(seed)
    cropped_edge = trans(edge)
    torch.random.manual_seed(seed)
    cropped_skeleton = trans(skeleton)
    torch.random.manual_seed(seed)
    cropped_weight = trans(weight)
    return cropped_img, cropped_label, cropped_edge, cropped_skeleton, cropped_weight


def random_crop_with_edge_skeleton(images, labels, edge, skeleton, aim_size):
    trans = transforms.Compose([transforms.RandomCrop(aim_size)])
    seed = torch.random.seed()

    torch.random.manual_seed(seed)
    cropped_img = trans(images)
    torch.random.manual_seed(seed)
    cropped_label = trans(labels)
    torch.random.manual_seed(seed)
    cropped_edge = trans(edge)
    torch.random.manual_seed(seed)
    cropped_skeleton = trans(skeleton)

    return cropped_img, cropped_label, cropped_edge, cropped_skeleton


def regular_crop(images, labels, aim_size):
    cropped_image = F.crop(images, 100, 100, 256, 256)
    cropped_label = F.crop(labels, 100, 100, 256, 256)
    return cropped_image, cropped_label


class ROAD_Dataset(Dataset):
    def __init__(self, txt, dataset_type, train, fractal_dir='', weight_dir='', edge_dir='', skeleton_dir='', decoder_type='',log_file='', epoch=0, update_d=5,img_size=256):

        self.img_size = img_size
        self.dataset_type = dataset_type
        self.train = train
        self.fractal_dir = fractal_dir
        self.weight_dir = weight_dir
        self.decoder_type = decoder_type
        self.log_file = log_file
        self.epoch = epoch
        self.update_d = update_d
        self.edge_dir = edge_dir
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
        if self.dataset_type == 'road':
            if self.train:
                img = cv2.imread(img_path, -1)
                img_ = img_PreProc_road(img, pro_type='norm_dataset')
                image_chw = np.transpose(img_, (2, 0, 1))
            else:
                test_path = img_path.replace('cropped_test_input', 'cropped_test_pre')
                image_chw = np.load(test_path).astype(np.float32)

            mask = cv2.imread(mask_path, 0)
            mask[mask > 3] = 255
            mask[mask <= 3] = 0

            image_chw_ = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            if self.train:
                cropped_img, cropped_mask = random_crop(image_chw_, mask_, 256)
            else:
                cropped_img = image_chw_
                cropped_mask = mask_
            sample = {"image": cropped_img,
                      "mask": cropped_mask,
                      "ID": os.path.split(img_path)[1]}
        elif self.dataset_type == 'road_copy':
            if self.train:
                npy_path = img_path.replace('train_val_aug', 'pre_train_val_aug')
                image_chw = np.load(npy_path[:-3] + 'npy').astype(np.float32)
            else:
                image_chw = np.load(img_path).astype(np.float32)

            mask = cv2.imread(mask_path, 0)
            mask[mask > 3] = 255
            mask[mask <= 3] = 0

            fractal_img = np.stack((image_chw[0], image_chw[1], image_chw[2], image_chw[0], image_chw[1], image_chw[2]),
                                   axis=2)
            image_chw2 = np.transpose(fractal_img, (2, 0, 1))

            image_chw_ = torch.from_numpy(image_chw2).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()
            if self.train:
                cropped_img, cropped_mask = random_crop(image_chw_, mask_, 256)
            else:
                cropped_img = image_chw_
                cropped_mask = mask_

            sample = {"image": cropped_img,
                      "mask": cropped_mask,
                      "ID": os.path.split(img_path)[1]}
        elif self.dataset_type == 'road_fractal':
            if self.train:
                npy_path = img_path.replace('train_val_aug', 'pre_train_val_aug')
                image_chw = np.load(npy_path[:-3] + 'npy').astype(np.float32)

            else:
                image_chw = np.load(img_path).astype(np.float32)

            mask = cv2.imread(mask_path, 0)
            mask[mask > 3] = 255
            mask[mask <= 3] = 0

            image_chw_ = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()
            if self.train:
                cropped_img, cropped_mask = random_crop(image_chw_, mask_, 256)
            else:
                cropped_img = image_chw
                cropped_mask = mask_

            sample = {"image": cropped_img,
                      "mask": cropped_mask,
                      "ID": os.path.split(img_path)[1]}
        elif self.dataset_type == 'road_fractal_three_decoder':
            if self.train:
                npy_path = img_path.replace('train_val_aug', 'pre_train_val_aug')
                image_chw = np.load(npy_path[:-3] + 'npy').astype(np.float32)
            else:
                image_chw = np.load(img_path).astype(np.float32)
            mask = cv2.imread(mask_path, 0)
            mask[mask > 3] = 255
            mask[mask <= 3] = 0
            image_chw_ = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()
            if self.train:
                edge_path = mask_path.replace('train_val_labels_aug', self.edge_dir)
                skeleton_path = mask_path.replace('train_val_labels_aug', self.skeleton_dir)
                edge = cv2.imread(edge_path, -1)
                skeleton = cv2.imread(skeleton_path, -1)
                skeleton_ = torch.from_numpy(skeleton / 255.).unsqueeze_(dim=0).float()
                edge_ = torch.from_numpy(edge / 255.).unsqueeze_(dim=0).float()
                cropped_img, cropped_mask, cropped_edge, cropped_skeleton = random_crop_with_edge_skeleton(image_chw_,
                                                                                                           mask_,
                                                                                                           edge_,
                                                                                                           skeleton_,
                                                                                                           256)
                sample = {"image": cropped_img,
                          "mask": cropped_mask,
                          "skeleton": cropped_skeleton,
                          "edge": cropped_edge,
                          "ID": os.path.split(img_path)[1]}
            else:
                cropped_img = image_chw
                cropped_mask = mask_
                sample = {"image": cropped_img,
                          "mask": cropped_mask,
                          "ID": os.path.split(img_path)[1]}
        elif self.dataset_type == 'road_fractal_three_decoder_weighted':
            train_imgpth_list = img_path.split('/')
            image_name = train_imgpth_list[-1]
            if self.train:
                npy_path = img_path.replace('train_val_aug', 'pre_train_val_aug')
                image_chw = np.load(npy_path[:-3] + 'npy').astype(np.float32)
            else:
                image_chw = np.load(img_path).astype(np.float32)
            mask = cv2.imread(mask_path, 0)
            mask[mask > 3] = 255
            mask[mask <= 3] = 0
            image_chw_ = torch.from_numpy(image_chw).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            weight_path = mask_path.replace('train_val_labels_aug', 'train_val_labels_aug' + self.weight_dir)
            weighted_npy = weight_path.replace(".tif", ".npy")
            weight1 = np.ones_like(img)
            weight2 = np.load(weighted_npy)
            weight = weight1 + (weight2 / np.max(weight2))


            if self.train:
                edge_path = mask_path.replace('train_val_labels_aug', self.edge_dir)
                skeleton_path = mask_path.replace('train_val_labels_aug', self.skeleton_dir)
                edge = cv2.imread(edge_path, -1)
                skeleton = cv2.imread(skeleton_path, -1)
                skeleton_ = torch.from_numpy(skeleton / 255.).unsqueeze_(dim=0).float()
                edge_ = torch.from_numpy(edge / 255.).unsqueeze_(dim=0).float()
                weight_ = torch.from_numpy(weight / 1.0).unsqueeze_(dim=0).float()
                cropped_img, cropped_mask, cropped_edge, cropped_skeleton, cropped_weight = random_crop_with_edge_skeleton_weight(
                    image_chw_, mask_,
                    edge_, skeleton_, weight_, 256)

                sample = {"image": cropped_img,
                          "mask": cropped_mask,
                          "skeleton": cropped_skeleton,
                          "edge": cropped_edge,
                          "weight": weight_,
                          "ID": os.path.split(img_path)[1]}
            else:
                cropped_img = image_chw
                cropped_mask = mask_
                sample = {"image": cropped_img,
                          "mask": cropped_mask,
                          "ID": os.path.split(img_path)[1]}

        return sample

    def __len__(self):
        return len(self.img_mask_paths)
