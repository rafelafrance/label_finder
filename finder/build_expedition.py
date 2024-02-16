#!/usr/bin/env python3
import argparse
import csv
import textwrap
from pathlib import Path

from util.pylib import log

from finder.pylib import sheet_util


def main():
    log.started()
    args = parse_args()

    args.expedition_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.expedition_dir / "manifest.csv"

    with csv_path.open("w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "reduced_by"])

        for sheet_path in sorted(args.sheet_dir.glob("*")):
            writer.writerow([sheet_path.name, args.reduce_by])
            sheet_image = sheet_util.get_sheet_image(sheet_path)

            if args.reduce_by > 1:
                sheet_image = sheet_image.reduce(args.reduce_by)

            exp_path = args.expedition_dir / sheet_path.name
            sheet_image.save(str(exp_path))

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Create an expedition for the label parser.""",
        ),
    )

    arg_parser.add_argument(
        "--sheet-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""The sheet images are in this directory.""",
    )

    arg_parser.add_argument(
        "--expedition-dir",
        required=True,
        type=Path,
        metavar="PATH",
        help="""Place expedition files in this directory.""",
    )

    arg_parser.add_argument(
        "--reduce-by",
        type=int,
        default=1,
        metavar="N",
        help="""Shrink images by this factor. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
