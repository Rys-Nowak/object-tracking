import numpy as np
import cv2


def create_initial_object_model(current_frame, previous_frames, thresh=20):
    diffs = np.array([cv2.absdiff(current_frame, frame)
                     for frame in previous_frames])
    diffs = np.where(np.abs(diffs) > thresh, diffs, np.zeros(diffs.shape))
    diffs = diffs.astype(bool)*1

    bottom_estimation = current_frame
    for diff in diffs:
        bottom_estimation = bottom_estimation * diff

    top_estimation = current_frame * \
        (cv2.bitwise_or(bottom_estimation.astype(bool)*1, diffs[-1]))

    return bottom_estimation.astype(np.uint8), top_estimation.astype(np.uint8)
