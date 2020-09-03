import cv2
import os

from src.image_funcs.image_processor import process_image, nicely_overlay_images
from src.image_funcs.image_rotations import calculate_most_appropriate_rotation
from src.image_funcs.image_io_consts import (
    OUTPUT_MASKS,
    OUTPUT_OVERLAY,
    OUTPUT_ROTATION,
    EXPECTED_MINIMUM_PIXEL_COUNT,
)
from src.utils.mask_creator import MaskCreator
from src.utils.logger import logger


class ImageInformationWrapper:
    def __init__(
        self,
        name: str,
        input_path: str,
        output_directory: str,
        base_rotation_image,
        overlay_output_directory: str = OUTPUT_OVERLAY,
    ):
        self.name = name
        self.input_path = input_path
        self.output_directory = output_directory
        self.base_rotation_image = base_rotation_image
        self.overlay_output_directory = overlay_output_directory

        self.loaded = False
        self.processed = False

    def get_output_path(self):
        return os.path.join(self.output_directory, OUTPUT_MASKS, self.name)

    def get_rotation_output_path(self):
        return (
            os.path.join(self.output_directory, OUTPUT_ROTATION, self.name)[:-3] + "txt"
        )

    def get_overlay_path(self):
        return os.path.join(self.output_directory, OUTPUT_OVERLAY, self.name)

    def load_img(self):
        self.img = cv2.imread(
            self.input_path,
            cv2.IMREAD_GRAYSCALE,
        )
        self.loaded = True
        return self

    def process_img(self):
        if not self.loaded:
            raise ValueError("Image must be processed to save.")

        self.processed_img = process_image(self.img, MaskCreator.MASK_SMALL)

        white_pixel_count = cv2.countNonZero(self.processed_img)
        if white_pixel_count < EXPECTED_MINIMUM_PIXEL_COUNT:
            logger.info(
                "{} result shape incorrect. Attempting with a more generous mask.".format(
                    self.name
                )
            )
            self.processed_img = process_image(self.img, MaskCreator.MASK_LARGE)
            white_pixel_count = cv2.countNonZero(self.processed_img)

        if white_pixel_count < EXPECTED_MINIMUM_PIXEL_COUNT:
            logger.warn("{} failed to produce proper shape.".format(self.name))

        self.processed = True
        return self

    def save_img(self):
        if not self.loaded:
            raise ValueError("Image must be loaded and processed to save.")
        if not self.processed:
            raise ValueError("Image must be processed to save.")

        cv2.imwrite(self.get_output_path(), self.processed_img)
        return self

    def calculate_and_save_rotation(self):
        if not self.loaded:
            raise ValueError("Image must be loaded and processed to calculate.")
        if not self.processed:
            raise ValueError("Image must be processed to calculate.")
        self.rotation_deg = calculate_most_appropriate_rotation(
            self.processed_img, self.base_rotation_image
        )

        with open(self.get_rotation_output_path(), "w") as f:
            f.write(str(self.rotation_deg))
        return self

    def create_overlay(self):
        if not self.loaded:
            raise ValueError("Image must be loaded and processed to overlay.")
        if not self.processed:
            raise ValueError("Image must be processed to overlay.")
        overlay = nicely_overlay_images(self.img, self.processed_img)
        cv2.imwrite(self.get_overlay_path(), overlay)
        return self
