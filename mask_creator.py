import cv2
import numpy as np


class MaskCreator:
    MASK = None

    @staticmethod
    def get_mask(pathToMask=None, reset=False, **kwargs):
        # generate mask only once (or import it)
        if reset:
            MaskCreator.MASK = None

        if pathToMask is None:
            if MaskCreator.MASK is None:
                MaskCreator.MASK = MaskCreator.create_mask(**kwargs)
        else:
            MaskCreator.MASK = cv2.imread(imagename, cv2.IMREAD_GRAYSCALE)
        return MaskCreator.MASK

    @staticmethod
    def create_mask(radius=370, shape=(896, 896)):
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
    def save_mask(path="mask.png"):
        MaskCreator.MASK = create_mask()
        cv2.imwrite(path, MaskCreator.MASK)


if __name__ == "__main__":
    MaskCreator.save_mask()
