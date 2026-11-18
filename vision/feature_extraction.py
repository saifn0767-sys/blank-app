"""
Feature extraction IDENTICAL to the CK+ training script
(landmarks 153 + regional HOG 2376 + spatial LBP 360 = 2889 features).

The training functions below are copied verbatim from the training script.
Only the entry point differs: a webcam frame is first reduced to a face crop
of 48x48 (the same kind of input as the CK+48 images), then it goes through
exactly the same pipeline used in training.
"""
import threading
from pathlib import Path

import cv2
import dlib
import numpy as np
from skimage.feature import hog, local_binary_pattern

FACE_SIZE = 96
DETECTOR_UPSCALE = 4
CK_IMAGE_SIZE = 48          # CK+48 images are 48x48 face crops
CROP_MARGIN = 0.10          # context added around the detected face box
EXPECTED_FEATURES = 2889

_DETECTOR = dlib.get_frontal_face_detector()
_LOCK = threading.Lock()    # dlib objects are not thread-safe


def load_predictor(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dlib predictor file not found: {path}")
    return dlib.shape_predictor(str(path))


# ---------------- copied from the training script ----------------

def align_face_and_landmarks(image, landmarks):
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    center = (float((left_eye[0] + right_eye[0]) / 2),
              float((left_eye[1] + right_eye[1]) / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = image.shape
    aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
    landmarks_h = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
    aligned_landmarks = landmarks_h @ rotation_matrix.T
    return aligned_image, aligned_landmarks.astype(np.float32)


def extract_geometric_features(landmarks):
    nose_tip = landmarks[30]
    left_eye_center = landmarks[36:42].mean(axis=0)
    right_eye_center = landmarks[42:48].mean(axis=0)
    eye_distance = np.linalg.norm(right_eye_center - left_eye_center) + 1e-6
    normalized_points = (landmarks - nose_tip) / eye_distance
    selected_points = normalized_points[17:68]
    distances_to_nose = np.linalg.norm(selected_points, axis=1)
    return np.concatenate([selected_points.flatten(),
                           distances_to_nose]).astype(np.float32)


def extract_hog_region(gray_face, points, target_size=(48, 32), pad=4):
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(gray_face.shape[1], int(x2) + pad + 1)
    y2 = min(gray_face.shape[0], int(y2) + pad + 1)
    roi = gray_face[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)
    features = hog(roi, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys",
                   feature_vector=True)
    return features.astype(np.float32)


def extract_spatial_lbp(gray_face, grid_size=(6, 6)):
    lbp = local_binary_pattern(gray_face, P=8, R=1, method="uniform")
    h, w = lbp.shape
    gh = h // grid_size[0]
    gw = w // grid_size[1]
    features = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            patch = lbp[row * gh:(row + 1) * gh, col * gw:(col + 1) * gw]
            histogram, _ = np.histogram(patch.ravel(),
                                        bins=np.arange(0, 11), range=(0, 10))
            histogram = histogram.astype(np.float32)
            histogram /= (histogram.sum() + 1e-7)
            features.append(histogram)
    return np.concatenate(features).astype(np.float32)


def _features_from_ck_style_image(image, predictor):
    """Body of the training `extract_all_features`, starting after imread.
    `image` is a uint8 grayscale face image. Returns (features, landmarks)
    where landmarks are in the coordinates of `image` (before alignment)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    large_image = cv2.resize(image, None, fx=DETECTOR_UPSCALE,
                             fy=DETECTOR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    faces = _DETECTOR(large_image, 1)
    if len(faces) > 0:
        face = max(faces, key=lambda r: r.width() * r.height())
    else:
        h, w = large_image.shape
        face = dlib.rectangle(0, 0, w - 1, h - 1)   # same fallback as training

    shape = predictor(large_image, face)
    landmarks = np.array([[shape.part(i).x / DETECTOR_UPSCALE,
                           shape.part(i).y / DETECTOR_UPSCALE]
                          for i in range(68)], dtype=np.float32)

    aligned_image, aligned_landmarks = align_face_and_landmarks(image, landmarks)

    min_xy = aligned_landmarks.min(axis=0)
    max_xy = aligned_landmarks.max(axis=0)
    margin_x = (max_xy[0] - min_xy[0]) * 0.08
    margin_y = (max_xy[1] - min_xy[1]) * 0.12
    x1 = max(0, int(np.floor(min_xy[0] - margin_x)))
    y1 = max(0, int(np.floor(min_xy[1] - margin_y)))
    x2 = min(aligned_image.shape[1], int(np.ceil(max_xy[0] + margin_x)) + 1)
    y2 = min(aligned_image.shape[0], int(np.ceil(max_xy[1] + margin_y)) + 1)
    if x2 <= x1 or y2 <= y1:
        return None, None

    face_image = aligned_image[y1:y2, x1:x2]
    if face_image.size == 0:
        return None, None

    crop_landmarks = aligned_landmarks.copy()
    crop_landmarks[:, 0] -= x1
    crop_landmarks[:, 1] -= y1
    old_h, old_w = face_image.shape
    face_image = cv2.resize(face_image, (FACE_SIZE, FACE_SIZE),
                            interpolation=cv2.INTER_CUBIC)
    crop_landmarks[:, 0] *= FACE_SIZE / max(old_w, 1)
    crop_landmarks[:, 1] *= FACE_SIZE / max(old_h, 1)
    crop_landmarks = np.clip(crop_landmarks, 0, FACE_SIZE - 1)

    landmark_features = extract_geometric_features(crop_landmarks)
    pts = crop_landmarks.astype(np.int32)

    left_brow_eye = np.vstack([pts[17:22], pts[36:42]])
    right_brow_eye = np.vstack([pts[22:27], pts[42:48]])
    nose_points = pts[27:36]
    mouth_points = pts[48:68]

    left_hog = extract_hog_region(face_image, left_brow_eye, target_size=(48, 32))
    right_hog = extract_hog_region(face_image, right_brow_eye, target_size=(48, 32))
    nose_hog = extract_hog_region(face_image, nose_points, target_size=(32, 48), pad=5)
    mouth_hog = extract_hog_region(face_image, mouth_points, target_size=(64, 32))
    if any(f is None for f in [left_hog, right_hog, nose_hog, mouth_hog]):
        return None, None

    hog_features = np.concatenate([left_hog, right_hog, nose_hog, mouth_hog])
    lbp_features = extract_spatial_lbp(face_image, grid_size=(6, 6))

    features = np.concatenate([landmark_features, hog_features,
                               lbp_features]).astype(np.float32)
    return features, landmarks


# ---------------- live-camera entry point ----------------

def extract_all_features(image_bgr, predictor):
    """Return (features row, 48x48 face crop, landmark points in frame coords).
    features is None when no face is found."""
    gray = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if image_bgr.ndim == 3 else image_bgr.copy())

    with _LOCK:
        faces = _DETECTOR(gray, 1)
        if len(faces) == 0:
            return None, None, None
        rect = max(faces, key=lambda r: r.width() * r.height())

        mx, my = int(rect.width() * CROP_MARGIN), int(rect.height() * CROP_MARGIN)
        x1, y1 = max(0, rect.left() - mx), max(0, rect.top() - my)
        x2 = min(gray.shape[1], rect.right() + mx)
        y2 = min(gray.shape[0], rect.bottom() + my)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, None

        face48 = cv2.resize(crop, (CK_IMAGE_SIZE, CK_IMAGE_SIZE),
                            interpolation=cv2.INTER_AREA)
        features, lm48 = _features_from_ck_style_image(face48, predictor)

    if features is None or not np.isfinite(features).all():
        return None, None, None

    sx = crop.shape[1] / CK_IMAGE_SIZE
    sy = crop.shape[0] / CK_IMAGE_SIZE
    points = np.stack([x1 + lm48[:, 0] * sx, y1 + lm48[:, 1] * sy], axis=1)
    return features.reshape(1, -1), face48, points.astype(np.float32)
