import cv2
from src.image_funcs.image_processor import calculate_image_similarity


def rotate_img(img, deg):
    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def calculate_most_appropriate_rotation(img, base):
    max_similarity = 0
    best_degree = 0
    early_stop = 0
    for degree_front, degree_back in [(deg, 359 - deg) for deg in range(180)]:
        # iterate both to the left and to the right, as most images are centered around 0
        if early_stop == 10:
            # stop cycle after 10 more degrees
            # if we don't see any improvements once some match found
            break

        similarity = calculate_image_similarity(base, rotate_img(img, degree_front))
        if similarity > max_similarity:
            max_similarity = similarity
            best_degree = degree_front
            early_stop = 0

        similarity = calculate_image_similarity(base, rotate_img(img, degree_back))
        if similarity > max_similarity:
            max_similarity = similarity
            best_degree = degree_back
            early_stop = 0

        elif max_similarity > 0:
            early_stop += 1
    return best_degree
