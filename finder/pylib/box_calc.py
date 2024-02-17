import numpy as np
import numpy.typing as npt


def find_box_groups(boxes: npt.NDArray, threshold: float = 0.8) -> npt.ArrayLike:
    """
    Find overlapping sets of bounding boxes.

    Args:
    ----
        boxes: A 2D array of box coordinates shaped like np.array(N, 4).
            Each box is given in left, top, right, bottom order.

        threshold: Only consider boxes to overlap if the Intersection over Union (IoU))
            is >= this value. The range is [0.0, 1.0]. A higher value means that fewer
            boxes will match.

    Returns:
    -------
        A 1D array of length N, that labels what group a box belongs to.

    Example:
    -------
        boxes = np.array(
            [
                [100, 100, 400, 400],  # Group 1
                [500, 500, 600, 600],  # ..... 2
                [510, 510, 610, 610],  # ..... 2
                [110, 110, 410, 410],  # ..... 1
                [490, 490, 590, 590],  # ..... 2
            ]
        )
        find_box_groups(boxes, 0.5) == [1, 2, 2, 1, 2]  # There are 2 label sets.
    """
    if len(boxes) == 0:
        return np.array([])

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float64")

    # Simplify access to box components
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    area = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)

    # Sort by area
    idx = area.argsort()

    # This holds the label set
    overlapping = np.zeros_like(idx)

    group = 0
    while len(idx) > 0:
        group += 1

        # Pop the largest box
        curr = idx[-1:]
        idx = idx[:-1]

        overlapping[curr] = group

        found = True  # Do we need to look for more overlapping boxes
        start = 0  # Used to skip repeated searches

        # Every time we find new matches we need to check the new ones against the rest
        while found:
            found = False

            for c in curr[start:]:
                # Get interior (overlap) coordinates
                xx0 = np.maximum(x0[c], x0[idx])
                yy0 = np.maximum(y0[c], y0[idx])
                xx1 = np.minimum(x1[c], x1[idx])
                yy1 = np.minimum(y1[c], y1[idx])

                # Get the intersection over the union (IOU) with the current box
                iou_ = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
                iou_ /= area[idx] + area[c] - iou_

                # Find IOUs larger than threshold & group them
                iou_ = np.where(iou_ >= threshold)[0]

                if len(iou_):
                    found = True
                    overlapping[idx[iou_]] = group  # Mark the found boxes
                    start = len(curr)  # Skip already searched boxes
                    curr = np.hstack((curr, idx[iou_]))  # Append to current indexes
                    idx = np.delete(idx, iou_)  # Remove all matching indexes

    return overlapping
