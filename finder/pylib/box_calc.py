import numpy as np


def find_box_groups(boxes: np.array, threshold: float = 0.8):
    """Find overlapping sets of bounding boxes."""
    if len(boxes) == 0:
        return np.array([])

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float64")

    # Simplify access to box components
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    area = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)

    idx = area
    idx = idx.argsort()

    overlapping = np.zeros_like(idx)
    group = 0
    while len(idx) > 0:
        group += 1

        # Pop the largest box
        curr = idx[-1:]
        idx = idx[:-1]

        overlapping[curr] = group

        found = True
        start = 0

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
                    overlapping[idx[iou_]] = group
                    start = len(curr)
                    curr = np.hstack((curr, idx[iou_]))  # Append to current indexes
                    idx = np.delete(idx, iou_)  # Remove all matching indexes

    return overlapping
