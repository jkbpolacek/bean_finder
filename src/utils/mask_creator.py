import cv2
import numpy as np
import os

from src.utils.logger import logger


SMALL_MASK_PATH = "../data/mask_small.png"
LARGE_MASK_PATH = "../data/mask_large.png"

SMALL_MASK_RADIUS = 370
LARGE_MASK_RADIUS = 380


class MaskCreator:
    MASK_SMALL = None
    MASK_LARGE = None

    @staticmethod
    def load_mask():
        logger.info("Loading masks.")
        if not os.path.isfile(SMALL_MASK_PATH) or not os.path.isfile(LARGE_MASK_PATH):
            logger.info("Mask files not found, generating them instead.")
            MaskCreator.init_mask()
        else:
            MaskCreator.MASK_SMALL = cv2.imread(
                SMALL_MASK_PATH,
                cv2.IMREAD_GRAYSCALE,
            )
            MaskCreator.MASK_LARGE = cv2.imread(
                LARGE_MASK_PATH,
                cv2.IMREAD_GRAYSCALE,
            )
            logger.info("Masks loaded.")

    @staticmethod
    def init_mask():
        logger.info("Generating mask.")
        MaskCreator.MASK_SMALL = MaskCreator.create_mask(SMALL_MASK_RADIUS)
        MaskCreator.MASK_LARGE = MaskCreator.create_mask(LARGE_MASK_RADIUS)

    @staticmethod
    def create_mask(radius, shape=(896, 896)):
        # won't work for odd shape dimensions
        core_mask = np.zeros(shape, dtype=np.int8)
        x_center = shape[0] // 2
        y_center = shape[1] // 2
        radiuspower2 = radius ** 2

        for x in range(shape[0]):
            for y in range(shape[1]):
                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radiuspower2:
                    core_mask[x][y] = 255
        return core_mask

    @staticmethod
    def save_mask(radius, path, shape=(896, 896)):
        mask = MaskCreator.create_mask(radius)
        cv2.imwrite(path, mask)


if __name__ == "__main__":
    MaskCreator.save_mask(SMALL_MASK_RADIUS, path=LARGE_MASK_PATH)
    MaskCreator.save_mask(LARGE_MASK_RADIUS, path=SMALL_MASK_PATH)
