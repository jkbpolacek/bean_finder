import os
import cv2
import numpy as np
import pandas as pd

from clean_image import dilate, preclean, repeat_median_blur
from mask_creator import MaskCreator


def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def get_filtered_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filter out only inner contours
    contours_n = [c for i, c in enumerate(contours) if hierarchy[0, i, 3] != -1]
    hierarchy_n = hierarchy[hierarchy[:, :, 3] != -1]
    hierarchy_n = np.expand_dims(hierarchy_n, axis=0)
    return contours_n, hierarchy_n


def apply_contours(shape, contours, hierarchy):
    # could use hierarchy to avoid later need to flood fill, but doesn't work always
    contour_result_img = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(
        contour_result_img,
        contours,
        contourIdx=-1,  # draw all
        color=(255, 255, 255),
        thickness=cv2.FILLED,
        #hierarchy=hierarchy,
        #maxLevel=2,
    )
    return contour_result_img


def extract_shape_by_contours(img, max_clean_attepmts=5):
    contours = []
    contours, hierarchy = get_filtered_contours(img)

    # We want 2 contours for the "bean" shape,
    # the inner and outer bean.
    # If we only get 1 contour, we have failed to obtain the outer bean.
    # Therefore, we must manipulate the original image further
    # This, however, leads to lower mask accuracy.
    while len(contours) <= 1 and max_clean_attepmts > 0:
        max_clean_attepmts -= 1
        img = dilate(img)
        contours, hierarchy = get_filtered_contours(img)

    return apply_contours(img.shape, contours, hierarchy)


def flood_fill(img):
    # post-processing, sometimes only the outer contour of bean gets drawn for some reason
    im_floodfill = img.copy().astype(np.uint8)

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    return img | im_floodfill_inv


def remove_unwanted_blotches_by_labelling(img):
    ret, markers = cv2.connectedComponents(img)
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
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(img.shape, dtype="uint8")
        labelMask[markers == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > maxNumPixels:
            mask = labelMask
            maxNumPixels = numPixels
        
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    return apply_mask(img, mask)

def post_process(img):
    eliminated_artifacts = repeat_median_blur(img)
    return remove_unwanted_blotches_by_labelling(eliminated_artifacts)
   


def process_image(img):
    masked_img = apply_mask(img, MaskCreator.get_mask())
    _, thresh = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned_thresh = preclean(thresh)
    extracted_shape = extract_shape_by_contours(cleaned_thresh)
    flood_filled_img = flood_fill(extracted_shape)
    post_processed_img = post_process(flood_filled_img)
    return post_processed_img


def process_image_verbose(img):
    # shows output of every step of image processing

    masked_img = apply_mask(img, MaskCreator.get_mask())

    cv2.imshow("masked_img", masked_img)

    _, thresh = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned_thresh = preclean(thresh)

    cv2.imshow("cleaned_thresh", cleaned_thresh)

    extracted_shape = extract_shape_by_contours(cleaned_thresh)

    cv2.imshow("extracted_shape", extracted_shape)

    flood_filled_img = flood_fill(extracted_shape)
    
    cv2.imshow("flood_filled_img", flood_filled_img)

    post_processed_img = post_process(flood_filled_img)
    
    cv2.imshow("post_processed_img", post_processed_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return post_processed_img


def nicely_overlay_images(base_img, detected_shape_img):
    base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    detected_shape_img = cv2.cvtColor(detected_shape_img, cv2.COLOR_GRAY2RGB)

    detected_shape_img[:, :, 0] = np.zeros(
        [detected_shape_img.shape[0], detected_shape_img.shape[1]]
    )
    detected_shape_img[:, :, 1] = np.zeros(
        [detected_shape_img.shape[0], detected_shape_img.shape[1]]
    )

    return cv2.addWeighted(base_img, 0.8, detected_shape_img, 0.2, 0)
