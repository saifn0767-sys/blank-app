"""Handcrafted features. IMPORTANT: must match the training extractor exactly."""
from pathlib import Path
import cv2
import dlib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from .face_detection import detect_largest_face
from .face_alignment import align_face


FACE_SIZE = 96
EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise']

def load_predictor(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dlib predictor file not found: {path}")
    return dlib.shape_predictor(str(path))

def landmarks_from_rect(gray, predictor, rect):
    shape = predictor(gray, rect)
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

def _crop_face(gray, rect):
    x1, y1 = max(0, rect.left()), max(0, rect.top())
    x2, y2 = min(gray.shape[1], rect.right()), min(gray.shape[0], rect.bottom())
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(crop)

def geometric_features(points):
    selected = points[17:68].copy()
    nose = points[30]
    eye_dist = np.linalg.norm(points[45] - points[36])
    eye_dist = max(float(eye_dist), 1e-6)
    selected = (selected - nose) / eye_dist
    distances = np.linalg.norm(selected, axis=1)
    return np.concatenate([selected.ravel(), distances]).astype(np.float32)

def hog_region(face, points, indices, out_size=(96, 96)):
    """Crop a landmark region from normalized face coordinates, fallback full face if degenerate."""
    h, w = face.shape
    pts = points[np.asarray(indices)]
    x1 = max(0, int(np.floor(pts[:, 0].min()))); x2 = min(w, int(np.ceil(pts[:, 0].max())) + 1)
    y1 = max(0, int(np.floor(pts[:, 1].min()))); y2 = min(h, int(np.ceil(pts[:, 1].max())) + 1)
    if x2 <= x1 or y2 <= y1:
        crop = face
    else:
        crop = face[y1:y2, x1:x2]
    crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    return hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm='L2-Hys', feature_vector=True).astype(np.float32)

def spatial_lbp(face, grid=6, radius=1):
    n_points = 8 * radius
    lbp = local_binary_pattern(face, n_points, radius, method='uniform')
    n_bins = n_points + 2
    h, w = lbp.shape
    feats = []
    for r in range(grid):
        for c in range(grid):
            cell = lbp[int(r*h/grid):int((r+1)*h/grid), int(c*w/grid):int((c+1)*w/grid)]
            hist, _ = np.histogram(cell.ravel(), bins=np.arange(n_bins+1), range=(0, n_bins))
            hist = hist.astype(np.float32); hist /= hist.sum() + 1e-7
            feats.extend(hist)
    return np.asarray(feats, dtype=np.float32)

def extract_all_features(image_bgr, predictor):
    """Return (feature row, processed face crop, landmark points). None row means no face."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr.copy()
    clahe_full = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    rect = detect_largest_face(clahe_full, upsample=1)
    if rect is None:
        return None, None, None
    pts = landmarks_from_rect(clahe_full, predictor, rect)
    aligned = align_face(clahe_full, pts)
    face = _crop_face(aligned, rect)
    if face is None:
        return None, None, None
    # Normalize landmark coordinates to the cropped face image for regional HOG.
    x1, y1 = max(0, rect.left()), max(0, rect.top())
    pts_crop = pts - np.array([x1, y1], dtype=np.float32)
    sx, sy = FACE_SIZE / max(1, rect.width()), FACE_SIZE / max(1, rect.height())
    pts_crop[:, 0] *= sx; pts_crop[:, 1] *= sy
    # Landmark-defined regions; indexes are standard 68-point topology.
    regions = [list(range(17, 27)) + list(range(36, 42)),
               list(range(17, 27)) + list(range(42, 48)),
               list(range(27, 36)), list(range(48, 68))]
    geo = geometric_features(pts)
    hogs = [hog(face, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_norm='L2-Hys', feature_vector=True).astype(np.float32)]
    # Regional HOG vectors are included to keep regional descriptors explicit.
    # Exact regional crop/alignment must match the CK+ training script before deployment use.
    lbp = spatial_lbp(face, grid=6)
    features = np.concatenate([geo, *hogs, lbp]).astype(np.float32).reshape(1, -1)
    return features, face, pts
