import logging
import warnings

from PIL import Image, UnidentifiedImageError

IMAGE_EXCEPTIONS = (
    UnidentifiedImageError,
    ValueError,
    TypeError,
    FileNotFoundError,
    OSError,
)


def to_yolo_image(path, yolo_images, yolo_size) -> tuple[int, int] | None:
    yolo = yolo_images / path.name

    image = sheet_image(path)
    if not image:
        return None

    try:
        resized = image.resize((yolo_size, yolo_size))
        resized.save(yolo)

    except IMAGE_EXCEPTIONS as err:
        msg = f"Could not prepare {path.name}: {err}"
        logging.exception(msg)
        return None

    return image.size


def sheet_image(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings

        try:
            image = Image.open(path).convert("RGB")

        except IMAGE_EXCEPTIONS as err:
            msg = f"Could not prepare {path.name}: {err}"
            logging.exception(msg)
            return None

        return image
