#!/usr/bin/env python3
import argparse
import csv
import textwrap
from pathlib import Path

from tqdm import tqdm
from util.pylib import log

from finder.pylib import sheet_util


def main():
    log.started()
    args = parse_args()

    args.yolo_images.mkdir(exist_ok=True, parents=True)

    with args.sheet_csv.open() as csv_file:
        reader = csv.DictReader(csv_file)
        sheets = [r["path"] for r in reader]

    for path in tqdm(sheets):
        path = Path(path)
        sheet_util.to_yolo_image(path, args.yolo_images, args.yolo_size)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Prepare label inference data into a format for YOLO model inference.
            Required CSV columns:
                * "path": A path to the herbarium sheet image.
            """,
        ),
    )

    arg_parser.add_argument(
        "--sheet-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A CSV file containing all of the herbarium sheets paths to feed to the
            YOLO model.""",
    )

    arg_parser.add_argument(
        "--yolo-images",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save YOLO formatted images to this directory.""",
    )

    arg_parser.add_argument(
        "--yolo-size",
        type=int,
        metavar="INT",
        default=640,
        help="""Resize images to this height & width in pixels. This must match the
            the image size used to train the model. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
