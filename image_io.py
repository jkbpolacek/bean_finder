import cv2
import numpy as np
import os
from typing import List

from image_information_wrapper import ImageInformationWrapper


IMAGES_PATH = "../"
OUTPUT_PATH = "../out/"
OVERLAYS_PATH = "../out/overlay/"

# names = [imagename for imagename in os.listdir(IMAGES_PATH + "/min/")]

names = [
    "1_2020-07-13_10-46-40.481.png"
]

wrappers: List[ImageInformationWrapper] = [
    ImageInformationWrapper(
        name,
        IMAGES_PATH,
        OUTPUT_PATH,
        also_create_nice_overlay_output=True,
        overlay_output_directory=OVERLAYS_PATH,
    )
    for name in names
]
for wrap in wrappers:
    # wrap.load_img().process_img().save_img()
    wrap.load_img().process_img_verbose().save_img()