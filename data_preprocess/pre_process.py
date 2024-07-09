import numpy as np
import cv2

def img_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

def img_clahe_cm(img):
    b,g,r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    output = cv2.merge((b,g,r))
    return output

def img_normalized(img):
    std = np.std(img)
    mean = np.mean(img)
    img_normalized = (img - mean) / (std + 1e-10)
    return img_normalized


def convert_16to8(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 255).astype(np.uint8)
    return img

def convert_8to16(img):
    img = (img - np.mean(img)) / np.std(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (img * 65535).astype(np.uint16)
    return img

def sober_filter(img):
    if img.dtype == "uint16":
        dx = np.array(cv2.Sobel(img, cv2.CV_32F, 1, 0))
        dy = np.array(cv2.Sobel(img, cv2.CV_32F, 0, 1))
    elif img.dtype == "uint8":
        dx = np.array(cv2.Sobel(img, cv2.CV_16S, 1, 0))
        dy = np.array(cv2.Sobel(img, cv2.CV_16S, 0, 1))
    dx = np.abs(dx)
    dy = np.abs(dy)
    edge = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
    return edge


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

def npy_PreProc(npy):
    img_FD = npy[0]
    img_FL = npy[1]
    FD_min = np.min(img_FD)
    FD_max = np.max(img_FD)
    img_FD = (img_FD - FD_min) / (FD_max - FD_min)

    FL_min = np.min(img_FL)
    FL_max = np.max(img_FL)
    img_FL = (img_FL - FL_min) / (FL_max - FL_min)
    sd_FD = standardization(img_FD)
    sd_FL = standardization(img_FL)
    return sd_FD, sd_FL