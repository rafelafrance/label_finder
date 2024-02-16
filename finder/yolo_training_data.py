#!/usr/bin/env python3
import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from util.pylib import log

from finder.pylib import const, sheet_util


def main():
    log.started()
    args = parse_args()

    args.yolo_images.mkdir(exist_ok=True)
    args.yolo_labels.mkdir(exist_ok=True)

    sheets = get_sheets(args.label_csv)

    for path, labels in tqdm(sheets.items()):
        path = Path(path)
        image_size = sheet_util.to_yolo_image(path, args.yolo_images, args.image_size)
        if image_size is not None:
            write_labels(args.yolo_labels, labels, image_size)

    log.finished()


def get_sheets(label_csv) -> dict[list[dict]]:
    with label_csv.open() as csv_file:
        reader = csv.DictReader(csv_file)
        sheets = defaultdict(list)
        for label in reader:
            stem = Path(label["path"]).stem
            if label["class"]:
                sheets[stem].append(label)
            elif stem not in sheets:
                sheets[stem] = []
    return sheets


def write_labels(text_path, labels, image_size):
    classes = [lb["class"] for lb in labels]
    boxes = np.array(
        [[lb["left"], lb["top"], lb["right"], lb["bottom"]] for lb in labels],
        dtype=np.float64,
    )
    width, height = image_size
    boxes = to_yolo_format(boxes, width, height)
    with text_path.open("w") as txt_file:
        for label_class, box in zip(classes, boxes, strict=False):
            label_class = const.CLASS2INT[label_class]
            bbox = np.array2string(box, formatter={"float_kind": lambda x: "%.6f" % x})
            line = f"{label_class} {bbox[1:-1]}\n"
            txt_file.write(line)


def to_yolo_format(bboxes, sheet_width, sheet_height):
    """
    Convert bounding boxes to YOLO format.

    center x, center y, width, height
    convert to fraction of the image size
    """
    boxes = np.empty_like(bboxes)

    boxes[:, 0] = (bboxes[:, 2] + bboxes[:, 0]) / 2.0 / sheet_width  # Center x
    boxes[:, 1] = (bboxes[:, 3] + bboxes[:, 1]) / 2.0 / sheet_height  # Center y
    boxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0] + 1) / sheet_width  # Box width
    boxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1] + 1) / sheet_height  # Box height

    return boxes


def parse_args():
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Prepare label training data into a format for YOLO model training.
            Required CSV columns:
                * "path": A path to the herbarium sheet image.
                * "class": The label class in text format.
                * "left": The label's left most pixel.
                * "top": The label's top most pixel.
                * "right": The label's right most pixel.
                * "bottom": The label's bottom most pixel.
            If the class is empty then the sheet has no labels. This is so YOLO
            still uses the sheet for training.
            """,
        ),
    )

    arg_parser.add_argument(
        "--label-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A CSV file containing all of the label information for
            training a YOLO model.""",
    )

    arg_parser.add_argument(
        "--yolo-images",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save YOLO formatted images to this directory.""",
    )

    arg_parser.add_argument(
        "--yolo-labels",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save YOLO formatted label information to this directory.""",
    )

    arg_parser.add_argument(
        "--yolo-size",
        type=int,
        metavar="INT",
        default=640,
        help="""Resize images to this height & width in pixels.
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
