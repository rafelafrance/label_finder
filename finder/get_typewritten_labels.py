#!/usr/bin/env python3
import argparse
import shutil
import textwrap
from pathlib import Path

from util.pylib import log


def main():
    log.started()
    args = parse_args()

    typewritten = by_class(args.label_dir)

    move_labels(typewritten, args.typewritten_dir)

    log.finished()


def by_class(label_dir):
    return [p for p in label_dir.glob("*") if p if p.stem.find("_Typewritten_") > -1]


def move_labels(typewritten, typewritten_dir):
    typewritten_dir.mkdir(exist_ok=True)

    for src in typewritten:
        dst = typewritten_dir / src.name
        shutil.move(src, dst)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description=textwrap.dedent("""Filter labels to only typewritten labels."""),
    )

    arg_parser.add_argument(
        "--label-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Directory containing label images.""",
    )

    arg_parser.add_argument(
        "--typewritten-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Move typewritten labels to this directory.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
