import cv2
import numpy as np

def align_face(gray, landmarks, output_size=None):
    """Align by eye centers. Uses eye-line angle and returns a warped grayscale frame."""
    left = landmarks[36:42].mean(axis=0)
    right = landmarks[42:48].mean(axis=0)
    delta = right - left
    angle = np.degrees(np.arctan2(delta[1], delta[0]))
    center = tuple(((left + right) / 2.0).astype(float))
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    size = output_size or (gray.shape[1], gray.shape[0])
    return cv2.warpAffine(gray, matrix, size, flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
