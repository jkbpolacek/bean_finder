import cv2
import os
from typing import List
from tqdm.contrib.concurrent import process_map


from src.image_funcs.image_information_wrapper import ImageInformationWrapper
from src.image_funcs.image_io_consts import (
    MAX,
    MEDIAN,
    MIN,
    OUTPUT_MASKS,
    OUTPUT_OVERLAY,
    OUTPUT_ROTATION,
)
from src.utils.logger import logger
from src.utils.mask_creator import MaskCreator


def process_wrapper(wrap):
    try:
        wrap.load_img().process_img().save_img().calculate_and_save_rotation()
    except BaseException:
        logger.error("Error processing wrap", exc_info=True)


def process_wrapper_overlay(wrap):
    try:
        wrap.load_img().process_img().save_img().calculate_and_save_rotation().create_overlay()
    except BaseException:
        logger.error("Error processing wrap", exc_info=True)


def check_or_create_output_directories(output_directory: str, overlay: bool):
    dirs = [
        os.path.join(output_directory, OUTPUT_MASKS),
        os.path.join(output_directory, OUTPUT_ROTATION),
    ]
    if overlay:
        dirs.append(os.path.join(output_directory, OUTPUT_OVERLAY))

    for output_dir in dirs:
        os.makedirs(output_dir, exist_ok=True)


def handle_image_io(
    input_directory: str,
    output_directory: str,
    base_rotation_image_path: str,
    process_num: int,
    overlay: bool,
    chunksize: int,
    data_sub_set: str = MIN,
):
    check_or_create_output_directories(output_directory, overlay)
    MaskCreator.load_mask()
    base_rotation = cv2.imread(base_rotation_image_path, cv2.IMREAD_GRAYSCALE)

    logger.info(
        "Looking for images in {}".format(os.path.join(input_directory, data_sub_set))
    )

    names_paths = [
        (imagename, os.path.join(input_directory, data_sub_set, imagename))
        for imagename in os.listdir(os.path.join(input_directory, data_sub_set))
    ]

    logger.info("{} images found.".format(len(names_paths)))
    wrappers: List[ImageInformationWrapper] = [
        ImageInformationWrapper(
            name=name_path[0],
            input_path=name_path[1],
            output_directory=output_directory,
            base_rotation_image=base_rotation,
        )
        for name_path in names_paths
    ]

    logger.info("Beginning image processing.")
    logger.info("Running in {} processes.".format(process_num))
    logger.info("Chunk size {}.".format(chunksize))

    if overlay:
        process_function = process_wrapper_overlay
    else:
        process_function = process_wrapper

    process_map(
        process_function,
        wrappers,
        chunksize=chunksize,
        max_workers=process_num,
    )

    logger.info("Processing finished.")
