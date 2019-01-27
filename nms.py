# refference
# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
import numpy as np

def non_max_suppression_slow(boxes, vconfidence, overlapThresh):
    if len(boxes) == 0:
        return []

    picked_idxs = []

    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    area = (x_max - x_min + 1) * (y_max - y_min + 1)
    # sorted by confidence
    idxs = np.argsort(vconfidence)[::-1]

    while len(idxs) > 0:
        last_idxs_index = len(idxs) - 1
        last_index = idxs[last_idxs_index]
        picked_idxs.append(last_index)
        suppress = [last_idxs_index]

        for current_index_index, current_index in enumerate(idxs[:-1]):
            x_min_max = max(x_min[last_index], x_min[current_index])
            y_min_max = max(y_min[last_index], y_min[current_index])
            x_max_min = min(x_max[last_index], x_max[current_index])
            y_max_min = min(y_max[last_index], y_max[current_index])

            width = max(0, x_max_min - x_min_max + 1)
            height = max(0, y_max_min - y_min_max + 1)

            intersection = float(width * height)
            overlap = intersection / (area[current_index] + area[last_index] - intersection)

            if overlap > overlapThresh:
                suppress.append(current_index_index)

        idxs = np.delete(idxs, suppress)
    return boxes[picked_idxs]