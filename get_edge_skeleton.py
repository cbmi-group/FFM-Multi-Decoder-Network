import numpy as np
import cv2
import itertools
import matplotlib.pyplot as plt
import os
from skimage import morphology

def edge_extract(root):
    img_root = os.path.join(root, 'masks')
    edge_root = os.path.join(root, 'masks_edges')

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)

    index = 0
    for name in file_names:
        img = cv2.imread(os.path.join(img_root, name), 0)
        edge, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img)
        cv2.drawContours(contour_img, edge, -1, (255), 1)
        cv2.imwrite(os.path.join(edge_root, name), contour_img)
        index += 1
    return 0



def skeleton_extract(root):
    img_root = os.path.join(root, 'train_val_labels_aug')
    skeleton_root = os.path.join(root, 'train_val_labels_aug_bone')
    if not os.path.exists(skeleton_root):
        os.mkdir(skeleton_root)

    file_names = os.listdir(img_root)
    for name in file_names:
        img = cv2.imread(os.path.join(img_root, name), -1)
        img[img <= 100] = 0
        img[img > 100] = 1
        skeleton0 = morphology.skeletonize(img)
        skeleton = skeleton0.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(skeleton_root, name), skeleton)

    return 0



if __name__ == '__main__':
    train_er = "./data/ER/train/"
    edge_extract(train_er)
    skeleton_extract(train_er)
