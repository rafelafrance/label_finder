#!/usr/bin/env python3
import argparse
import csv
import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from util.pylib import log

from finder.pylib import box_calc as calc
from finder.pylib.const import CLASSES, OTHER, TYPEWRITTEN


@dataclass
class Sheet:
    boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.int32))
    types: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.str_))

    @staticmethod
    def bbox_from_json(coords: str) -> npt.ArrayLike:
        raw = json.loads(coords)
        box = np.array([raw["left"], raw["top"], raw["right"], raw["bottom"]])
        return box


def main():
    log.started()
    args = parse_args()

    sheets = get_sheet_boxes(
        args.unreconciled, args.sheet_column, args.box_columns, args.class_columns
    )

    if args.limit:
        sheets = dict(list(sheets.items())[: args.limit])

    reconciled = []

    for sheet_id, sheet in tqdm(sheets.items()):
        groups = calc.find_box_groups(sheet.boxes, args.iou_threshold)

        for grp in np.unique(groups):
            label = {"sheet": sheet_id}
            label |= merge_boxes(sheet.boxes[groups == grp], args.expand_by)
            label |= merge_types(sheet.types[groups == grp])

            reconciled.append(label)

    df = pd.DataFrame(reconciled)
    df.to_csv(args.reconciled, index=False)

    log.finished()


def merge_boxes(boxes: npt.NDArray, expand_by) -> dict[str, float]:
    """Get the outside dimensions of the boxes."""
    return {
        "left": np.min(boxes[:, 0]) * expand_by,
        "top": np.min(boxes[:, 1]) * expand_by,
        "right": np.max(boxes[:, 2]) * expand_by,
        "bottom": np.max(boxes[:, 3]) * expand_by,
    }


def merge_types(types: npt.ArrayLike) -> dict[str, str]:
    """Get the most common type."""
    counts = {c: 0 for c in CLASSES}
    for val in types:
        counts[val] += 1
    cls: str = OTHER if counts[OTHER] > counts[TYPEWRITTEN] else TYPEWRITTEN
    return {"class": cls}


def get_sheet_boxes(unreconciled, sheet_column, box_columns, class_columns):
    sheets = defaultdict(Sheet)

    with unreconciled.open() as unrec:
        reader = csv.DictReader(unrec)
        for row in reader:
            sheet_id = row[sheet_column]

            coords = [v for k, v in row.items() if k.startswith(box_columns)]
            boxes = np.array([Sheet.bbox_from_json(c) for c in coords if c])
            if len(boxes) == 0:
                continue

            sheets[sheet_id].boxes = np.vstack((sheets[sheet_id].boxes, boxes))

            types = [v for k, v in row.items() if k.startswith(class_columns)]
            types = [v if v == TYPEWRITTEN else OTHER for v in types[: len(boxes)]]
            sheets[sheet_id].types = np.hstack((sheets[sheet_id].types, types))

    return sheets


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description=textwrap.dedent(
            """
            Reconcile data from a "Label Babel" expedition.

            We need training data for the label finder model and we use use volunteers
            to build the initial batch of training data. That is, we use a "Notes from
            Nature" Zooniverse expedition to have volunteers (often 3 or more) draw all
            label bounding boxes around every label. Every volunteer draws a slightly
            different bounding box, so we use this script to reconcile the differences
            into a single "best" label.
            """
        ),
    )

    arg_parser.add_argument(
        "--unreconciled",
        required=True,
        type=Path,
        metavar="PATH",
        help="""Get volunteer drawn labels from this CSV file. This is the CSV file
            gotten from the label_reconciliations.py script's --unreconciled option.""",
    )

    arg_parser.add_argument(
        "--reconciled",
        required=True,
        type=Path,
        metavar="PATH",
        help="""Write reconciled labels to this CSV file.""",
    )

    arg_parser.add_argument(
        "--expand-by",
        type=int,
        default=1,
        metavar="N",
        help="""The sheet images were reduced by this factor when submitting the
            expedition so we need to expand the results by the same factor to get back
            to the original size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.6,
        help="""Consider 2 boxes to be overlapping if the IoU is at least this value.
            A number between 0.0 and 1.0. Lower means more boxes will match and higher
            means fewer boxes will match. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--sheet-column",
        metavar="NAME",
        default="subject_Filename",
        help="""This column in the column in the --unreconciled CSV that holds the
            sheet image file name. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--box-columns",
        metavar="PATTERN",
        default="Box(es): box #",
        help="""The box coordinates are in columns that start with this pattern.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--class-columns",
        metavar="PATTERN",
        default="Box(es): select #",
        help="""The box types are in columns that start with this pattern.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        help="""Sample this many sheets. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
