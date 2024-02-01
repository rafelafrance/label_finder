import csv
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from finder.pylib import const, sheet_util


def build(args: Namespace) -> None:
    args.yolo_images.mkdir(exist_ok=True)
    args.yolo_labels.mkdir(exist_ok=True)

    sheets = get_sheets(args.label_csv)

    for path, labels in tqdm(sheets.items()):
        path = Path(path)
        image_size = sheet_util.to_yolo_image(path, args.yolo_images, args.image_size)
        if image_size is not None:
            write_labels(args.yolo_labels, labels, image_size)


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
