#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from tqdm import tqdm
from util.pylib import log

from finder.pylib import const, sheet_util


def main():
    log.started()
    args = parse_args()
    to_labels(args)
    log.finished()


def to_labels(args):
    args.label_dir.mkdir(exist_ok=True, parents=True)

    sheet_paths = {p.stem: p for p in args.sheet_dir.glob("*")}

    label_paths = sorted(args.yolo_labels.glob("*.txt"))
    for label_path in tqdm(label_paths):
        sheet_path = sheet_paths.get(label_path.stem)
        if not sheet_path:
            continue

        sheet_image = sheet_util.get_sheet_image(sheet_path)

        with label_path.open() as lb:
            lines = lb.readlines()

        stem = label_path.stem

        for ln in lines:
            cls, left, top, right, bottom = from_yolo_format(ln, sheet_image)

            name = "_".join([stem, cls, str(left), str(top), str(right), str(bottom)])
            name += sheet_path.suffix

            label_image = sheet_image.crop((left, top, right, bottom))
            label_image.save(args.label_dir / name)


def from_yolo_format(ln, sheet_image):
    """Convert YOLO coordinates to image coordinates."""
    cls, center_x, center_y, width, height, *_ = ln.split()

    cls = const.CLASS2NAME[int(cls)]

    # Scale from fractional to sheet image size
    sheet_width, sheet_height = sheet_image.size
    center_x = float(center_x) * sheet_width
    center_y = float(center_y) * sheet_height
    radius_x = float(width) * sheet_width / 2
    radius_y = float(height) * sheet_height / 2

    # Calculate label's pixel coordinates
    left = round(center_x - radius_x)
    top = round(center_y - radius_y)
    right = round(center_x + radius_x)
    bottom = round(center_y + radius_y)

    return cls, left, top, right, bottom


def parse_args():
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """This script converts YOLO results to label images.""",
        ),
    )

    arg_parser.add_argument(
        "--yolo-labels",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Directory containing the label predictions.""",
    )

    arg_parser.add_argument(
        "--sheet-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""The directory containing all of the original herbarium sheet images.""",
    )

    arg_parser.add_argument(
        "--label-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Output the label images to this directory.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
