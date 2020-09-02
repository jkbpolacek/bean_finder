import numpy as np
import cv2

MEDIAN_BLUR_CONST_SOFT = 3
MEDIAN_BLUR_CONST_HARD = 5
KERNEL3x3 = np.ones((3, 3), np.int8)
KERNEL5x5 = np.ones((5, 5), np.int8)
KERNEL7x7 = np.ones((7, 7), np.int8)


def preclean(img, kernel=KERNEL7x7, median_ksize=MEDIAN_BLUR_CONST_SOFT):
    img = cv2.medianBlur(img, median_ksize)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def dilate(img, kernel=KERNEL3x3):
    return cv2.dilate(img, kernel, iterations=1)


def repeat_median_blur(img, iterations=100, median_ksize=MEDIAN_BLUR_CONST_HARD):
    for _ in range(iterations):
        img = cv2.medianBlur(img, median_ksize)
    return img
