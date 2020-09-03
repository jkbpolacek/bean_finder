import numpy as np
import cv2

MEDIAN_BLUR_CONST_SOFT = 3
MEDIAN_BLUR_CONST_HARD = 5
KERNEL3x3 = np.ones((3, 3), np.int8)
KERNEL5x5 = np.ones((5, 5), np.int8)
KERNEL7x7 = np.ones((7, 7), np.int8)


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def preclean(img, kernel=KERNEL7x7, median_ksize=MEDIAN_BLUR_CONST_SOFT):
    img = cv2.medianBlur(img, median_ksize)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def dilate(img, kernel=KERNEL3x3):
    return cv2.dilate(img, kernel, iterations=1)


def repeat_median_blur(img, iterations=100, median_ksize=MEDIAN_BLUR_CONST_HARD):
    for _ in range(iterations):
        img = cv2.medianBlur(img, median_ksize)
    return img


def remove_unwanted_blotches_by_labelling(img):
    _, markers = cv2.connectedComponents(img)
    mask = None
    maxNumPixels = 0
    labels = np.unique(markers)

    if len(labels) <= 2:
        # only 1 area detected
        return img

    for label in labels:
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and
        # count the number of pixels
        labelMask = np.zeros(img.shape, dtype="uint8")
        labelMask[markers == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > maxNumPixels:
            mask = labelMask
            maxNumPixels = numPixels

    return apply_mask(img, mask)
