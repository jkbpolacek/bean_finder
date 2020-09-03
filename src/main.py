#!/usr/bin/env python3
import argparse
import os

from src.image_funcs.image_io import handle_image_io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        "-i",
        type=str,
        help="Path to input directory.",
        required=True,
    )
    parser.add_argument(
        "--outputs",
        "-o",
        type=str,
        help="Path to output directory.",
        required=True,
    )
    parser.add_argument(
        "--create_overlay",
        default=False,
        action="store_true",
        help="Create visual images of masks overlaid with original images.",
    )
    parser.add_argument(
        "--rotation_default",
        "-r",
        type=str,
        help="Path to rotation comparator.",
        default="data/base_rotation.png",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="Amount of processes (threads) to run in.",
        default=4,
    )
    parser.add_argument(
        "--chunk",
        "-c",
        type=int,
        help="Chunk size for processing.",
        default=10,
    )

    args = parser.parse_args()

    if not os.path.isdir(args.inputs):
        raise ValueError("Input directory {} does not exist.".format(args.inputs))

    if not os.path.isdir(args.outputs):
        raise ValueError("Output directory {} does not exist.".format(args.outputs))

    if not os.path.isfile(args.rotation_default):
        raise ValueError(
            "Rotation default {} does not exist.".format(args.rotation_default)
        )

    handle_image_io(
        input_directory=args.inputs,
        output_directory=args.outputs,
        base_rotation_image_path=args.rotation_default,
        process_num=args.processes,
        chunksize=args.chunk,
        overlay=args.create_overlay,
    )


if __name__ == "__main__":
    main()
