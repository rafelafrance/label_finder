#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

from util.pylib import log

from finder.pylib.yolo import yolo_to_labels


def main():
    log.started()
    args = parse_args()
    yolo_to_labels.to_labels(args)
    log.finished()


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
