import csv
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

from finder.pylib import sheet_util


def build(args: Namespace) -> None:
    args.yolo_images.mkdirs(exist_ok=True)

    with args.sheet_csv.open() as csv_file:
        reader = csv.DictReader(csv_file)
        sheets = [r["path"] for r in reader]

    for path in tqdm(sheets):
        path = Path(path)
        sheet_util.to_yolo_image(path, args.yolo_images, args.yolo_size)
