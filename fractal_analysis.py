import numpy as np
import os
import cv2
import csv
import math
import os
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import torch

'''
The function is a modified box-counting algorithm to compute Fractal Dimension for image, as described by Wen-Li Lee and Kai-Sheng Hsieh.

Input:
    image: A 2D array containing a grayscale image;
Output:
    D: fractal dimension of image, as estimated by the modified box-counting algorithm;
'''


def Box_counting_modified(image):
    M = image.shape[0]  # image shape
    G_min = image.min()  # lowest gray level (0=white)
    G_max = image.max()  # highest gray level (255=black)
    G = G_max - G_min + 1  # number of gray levels, typically 256
    prev = -1  # used to check for plateaus
    r_Nr = []

    for L in range(2, (M // 2) + 1):
        h = max(1, G // (M // L))  # minimum box height is 1
        N_r = 0
        r = L / M
        for i in range(0, M, L):
            boxes = [[]] * ((G + h - 1) // h)  # create enough boxes with height h to fill the fractal space
            for row in image[i:i + L]:  # boxes that exceed bounds are shrunk to fit
                for pixel in row[i:i + L]:
                    height = (pixel - G_min) // h  # lowest box is at G_min and each is h gray levels tall
                    boxes[height].append(pixel)  # assign the pixel intensity to the correct box
            stddev = np.sqrt(np.var(boxes, axis=1))  # calculate the standard deviation of each box
            stddev = stddev[~np.isnan(stddev)]  # remove boxes with NaN standard deviations (empty)
            nBox_r = 2 * (stddev // h) + 1
            N_r += sum(nBox_r)
        if N_r != prev:  # check for plateauing
            r_Nr.append([r, N_r])
            prev = N_r
    x = np.array([np.log(1 / point[0]) for point in r_Nr])  # log(1/r)
    y = np.array([np.log(point[1]) for point in r_Nr])  # log(Nr)
    D = np.polyfit(x, y, 1)  # D = lim r -> 0 log(Nr)/log(1/r)
    return D


'''
The function to compute Fractal Feature Map for image.

Input:
    image: A 2D array containing a grayscale image;;
    window_size: the size of sliding window;
    step_size: the size of sliding step;
Output:
    FFM: the fractal feature map of image.

'''


def compute_FFM(image, step_size, window_size):
    img_shape = np.shape(image)
    result_x = math.ceil(img_shape[0] / step_size)
    result_y = math.ceil(img_shape[1] / step_size)
    FD = np.zeros((result_x, result_y))
    Length = np.zeros((result_x, result_y))
    H = img_shape[0]
    pad_size = math.floor(window_size / 2)
    padded_img = np.pad(image, ((pad_size, pad_size)), 'linear_ramp')
    for i in range(0, H, step_size):
        for j in range(0, H, step_size):
            selected_img = padded_img[i:i + window_size, j:j + window_size]
            selected_img_info = Box_counting_modified(selected_img)
            save_coor_x = int(i / step_size)
            save_coor_y = int(j / step_size)
            FD[save_coor_x, save_coor_y] = selected_img_info[0]
            Length[save_coor_x, save_coor_y] = selected_img_info[1]
    FFM = np.zeros((2, result_x, result_y))
    FFM[0] = FD
    FFM[1] = Length
    return FFM

'''
The function to compute Fractal Feature Map for images using Pool

Input:
    file_path: the root path of images;
    window_size: the size of sliding window;
    step_size: the size of sliding step;
'''


def compute_FMM_Pool(file_path, window_size, step_size):
    save_path = file_path[:-1] + '_Fractal_info_' + str(window_size) + '_' + str(step_size) + '/'
    if step_size > 1:
        up_save_path = file_path[:-1] + '_Fractal_info_' + str(window_size) + '_' + str(step_size) + '_up/'
        os.makedirs(up_save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    filenames = os.listdir(file_path)
    img_num = len(filenames)
    images_fractal = []
    pool = Pool(16)
    for i in range(img_num):
        now_img = (cv2.imread(os.path.join(file_path, filenames[i]), 0)).astype(np.uint8)
        images_fractal.append(
            pool.apply_async(compute_FFM, args=(now_img, window_size, step_size,)))
    pool.close()
    pool.join()
    now_img = (cv2.imread(os.path.join(file_path, filenames[i]), 0)).astype(np.uint8)
    img_shape = np.shape(now_img)
    m = torch.nn.Upsample(size=img_shape, mode='bilinear')
    for if_index in range(img_num):
        image_weight_now = images_fractal[if_index].get()
        img_ = torch.from_numpy(image_weight_now).unsqueeze(dim=0).float()
        up_now_file = m(img_)
        up_now_npy = up_now_file.squeeze().numpy()
        if step_size > 1:
            np.save(os.path.join(up_save_path, filenames[if_index].split('.')[0] + '.npy'), up_now_npy)
        np.save(os.path.join(save_path, filenames[if_index].split('.')[0] + '.npy'), image_weight_now)
