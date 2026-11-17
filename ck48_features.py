"""
CK+48 Facial Emotion Recognition (Subject-Independent, 5-Fold CV)
Feature extraction: Normalized 68 landmarks + Left/Right Brow-Eye, Nose & Mouth HOG + Spatial 6x6 LBP
Dimensionality reduction: StandardScaler + PCA
Classifier: Logistic Regression (C=0.5)
No CNN is used.

Expected folder structure:
project/
├── ck48_features.py
├── shape_predictor_68_face_landmarks.dat
└── CK+48/
    ├── anger/
    ├── contempt/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    └── surprise/
"""

import bz2
import urllib.request
from pathlib import Path

import cv2
import dlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "CK+48"
OUTPUT_DIR = DATASET_DIR / "Extracted_Features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "emotion_model.joblib"

RANDOM_STATE = 42
N_SPLITS = 5
PCA_VARIANCE = 0.95
LOGREG_C = 0.5  # Regularization strength to prevent overfitting on high-dim features

EMOTIONS = ["happy", "sad", "angry", "surprised", "neutral", "fear", "disgust"]

DATASET_EMOTION_MAP = {
    "anger": "angry",
    "contempt": "neutral",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprised",
    "neutral": "neutral",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

app = Flask(__name__)
MODEL = None


def load_app_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(MODEL_PATH)
    return MODEL


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Facial Emotion Recognition</title>
<style>
*{box-sizing:border-box} body{margin:0;font-family:Arial,sans-serif;background:#f1f5f9;color:#172033;padding:24px} main{max-width:1000px;margin:auto;background:white;padding:24px;border-radius:16px;box-shadow:0 8px 30px #0f172a12} h1{margin-top:0}.layout{display:grid;grid-template-columns:1.2fr 1fr;gap:24px} video{width:100%;background:#111;border-radius:12px;min-height:220px;transform:scaleX(-1)} .controls{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0} input[type=text]{padding:12px;border:1px solid #cbd5e1;border-radius:8px;width:100%;margin:8px 0 14px} button{padding:12px 18px;border:0;border-radius:8px;background:#2563eb;color:white;font-weight:bold;cursor:pointer} button.stop{background:#475569} button:disabled{opacity:.45;cursor:not-allowed} .status{padding:10px;border-radius:8px;background:#f1f5f9;margin:12px 0} .on{color:#15803d;font-weight:bold}.off{color:#b91c1c;font-weight:bold}.score{margin:12px 0}.score-head{display:flex;justify-content:space-between;font-size:14px;margin-bottom:5px}.track{height:10px;background:#e2e8f0;border-radius:20px;overflow:hidden}.fill{height:100%;background:#2563eb}.result{padding:18px;background:#eff6ff;border-radius:12px;margin:12px 0}.dominant{font-size:28px;font-weight:bold;color:#166534}.small{color:#64748b;font-size:13px}@media(max-width:750px){.layout{grid-template-columns:1fr}body{padding:10px}main{padding:16px}}
</style>
</head>
<body>
<main>
<h1>Facial Emotion Recognition</h1>
<p class="small">Traditional computer vision features + Multinomial Logistic Regression</p>
<div class="layout">
<section>
<div id="cameraStatus" class="status off">CAMERA OFF</div>
<video id="video" autoplay muted playsinline></video>
<label for="personId"><b>Person ID</b></label>
<input id="personId" type="text" maxlength="100" placeholder="Example: P001">
<label><input type="checkbox" id="consent"> I consent to webcam-based facial analysis.</label>
<div class="controls">
<button id="startBtn">Start Recognition</button>
<button id="stopBtn" class="stop" disabled>Stop</button>
</div>
<div id="status" class="status">Please provide consent before starting.</div>
</section>
<section>
<h2>Recognition Result</h2>
<div class="result">
<div class="small">DOMINANT EMOTION</div>
<div id="dominant" class="dominant">Waiting...</div>
<div id="confidence">Confidence: --</div>
</div>
<h3>Emotion probabilities</h3>
<div id="scores"><p class="small">No prediction yet.</p></div>
</section>
</div>
<p class="small">Privacy: the camera is activated only after consent. Each frame is sent to the local Flask API for prediction and processed in memory. Images are not saved; only prediction scores and the person ID are stored.</p>
</main>
<script>
const API_URL = "/api/predict";
const video = document.getElementById("video");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const consent = document.getElementById("consent");
const personInput = document.getElementById("personId");
const statusBox = document.getElementById("status");
const cameraStatus = document.getElementById("cameraStatus");
let stream = null; let timer = null; let running = false; let busy = false;
startBtn.addEventListener("click", startRecognition);
stopBtn.addEventListener("click", stopRecognition);
async function startRecognition(){
  if(!consent.checked){ alert("Please provide consent first."); return; }
  if(!personInput.value.trim()){ alert("Please enter a Person ID."); return; }
  startBtn.disabled = true;
  try{
    const health = await fetch("/api/health");
    if(!health.ok) throw new Error("Backend is unavailable.");
    const healthData = await health.json();
    if(!healthData.model_loaded) throw new Error("Trained model is not loaded.");
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:"user"}, audio:false});
    video.srcObject = stream; await video.play();
    running = true; cameraStatus.textContent = "CAMERA ON"; cameraStatus.className = "status on"; statusBox.textContent = "Recognition started. Capturing every 2 seconds."; stopBtn.disabled = false;
    captureAndPredict(); timer = setInterval(captureAndPredict, 2000);
  }catch(error){
    statusBox.textContent = error.message || "Could not start camera."; stopRecognition();
  }finally{ startBtn.disabled = false; }
}
async function captureAndPredict(){
  if(!running || busy || !video.videoWidth) return; busy = true;
  try{
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.85));
    if(!blob) throw new Error("Could not capture camera frame.");
    const form = new FormData(); form.append("person_id", personInput.value.trim()); form.append("image", blob, "frame.jpg");
    const response = await fetch(API_URL, {method:"POST", body:form});
    const data = await response.json();
    if(!response.ok) throw new Error(data.error || "Prediction failed.");
    showPrediction(data); statusBox.textContent = "Saved prediction at " + new Date().toLocaleTimeString();
  }catch(error){ statusBox.textContent = error.message; }
  finally{ busy = false; }
}
function showPrediction(data){
  document.getElementById("dominant").textContent = data.dominant;
  document.getElementById("confidence").textContent = "Confidence: " + (data.confidence * 100).toFixed(2) + "%";
  const target = document.getElementById("scores"); target.innerHTML = "";
  for(const [emotion, score] of Object.entries(data.scores)){
    const pct = (score * 100).toFixed(2);
    const row = document.createElement("div"); row.className = "score";
    const head = document.createElement("div"); head.className = "score-head";
    const name = document.createElement("strong"); name.textContent = emotion;
    const value = document.createElement("span"); value.textContent = pct + "%";
    head.append(name, value);
    const track = document.createElement("div"); track.className = "track";
    const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = pct + "%"; track.appendChild(fill);
    row.append(head, track); target.appendChild(row);
  }
}
function stopRecognition(){
  running = false; if(timer){ clearInterval(timer); timer = null; }
  if(stream){ stream.getTracks().forEach(track => track.stop()); stream = null; }
  video.srcObject = null; cameraStatus.textContent = "CAMERA OFF"; cameraStatus.className = "status off"; stopBtn.disabled = true; startBtn.disabled = false; if(statusBox.textContent === "Recognition started. Capturing every 2 seconds.") statusBox.textContent = "Recognition stopped.";
}
window.addEventListener("beforeunload", stopRecognition);
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTML_PAGE


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists() and load_app_model() is not None})


@app.post("/api/predict")
def predict():
    if not MODEL_PATH.exists():
        return jsonify({"error": "Model not found. Train the model first with python train.py --train."}), 503

    model = load_app_model()
    person_id = request.form.get("person_id", "").strip()
    uploaded = request.files.get("image")
    if not person_id or len(person_id) > 100:
        return jsonify({"error": "A valid person_id (1-100 characters) is required."}), 400
    if uploaded is None:
        return jsonify({"error": "Image frame is required for prediction."}), 400

    raw = uploaded.read()
    if not raw:
        return jsonify({"error": "Invalid image or empty upload."}), 400

    try:
        image_array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Uploaded data is not a valid image."}), 400

        result = predict_image_path_from_array(image, model)
        return jsonify({
            "person_id": person_id,
            "dominant": result["dominant"],
            "confidence": result["confidence"],
            "scores": result["scores"],
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check server logs."}), 500


# ------------------------------------------------------------
# 2. FIND THE DIRECTORY THAT CONTAINS EMOTION FOLDERS
# ------------------------------------------------------------

def find_image_root(dataset_dir):
    """Find the directory whose immediate subfolders are emotion classes."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Place the CK+48 folder beside this Python script."
        )

    candidates = [dataset_dir] + [
        p for p in dataset_dir.rglob("*") if p.is_dir()
    ]

    for candidate in candidates:
        folder_names = {p.name.strip().lower() for p in candidate.iterdir() if p.is_dir()}
        if len(folder_names.intersection(EMOTIONS)) >= 2:
            return candidate

    raise FileNotFoundError(
        "Could not find emotion folders inside CK+48. "
        "Expected folders such as anger, contempt, disgust, fear, happy, sad, surprise."
    )


try:
    IMAGE_ROOT = find_image_root(DATASET_DIR)
except FileNotFoundError:
    IMAGE_ROOT = DATASET_DIR


# ------------------------------------------------------------
# 3. LOAD IMAGE PATHS AND LABELS
# ------------------------------------------------------------

def load_dataset(image_root):
    image_paths = []
    labels = []

    print("\n========== LOADING CK+48 ==========")

    for dataset_emotion in DATASET_EMOTION_MAP:
        class_dir = image_root / dataset_emotion

        if not class_dir.is_dir():
            print(f"Warning: folder not found: {dataset_emotion}")
            continue

        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        mapped_label = DATASET_EMOTION_MAP[dataset_emotion]
        print(f"{dataset_emotion}: {len(files)} images -> {mapped_label}")

        for path in files:
            image_paths.append(str(path))
            labels.append(mapped_label)

    if not image_paths:
        raise ValueError(
            f"No images found under {image_root}. "
            "Check that each emotion folder contains image files."
        )

    counts = pd.Series(labels).value_counts()
    print("Total images:", len(image_paths))
    print("\nClass distribution:\n", counts)

    return image_paths, np.asarray(labels)


# ------------------------------------------------------------
# 4. INITIALIZE DLIB 68-POINT LANDMARK PREDICTOR
# ------------------------------------------------------------

PREDICTOR_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"


def prepare_predictor():
    """Download and extract the official dlib 68-point predictor if needed."""
    if not PREDICTOR_PATH.exists():
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = Path(str(PREDICTOR_PATH) + ".bz2")

        print("Downloading dlib 68-landmark predictor...")
        urllib.request.urlretrieve(url, compressed_path)

        with bz2.open(compressed_path, "rb") as source:
            with open(PREDICTOR_PATH, "wb") as destination:
                destination.write(source.read())

        compressed_path.unlink(missing_ok=True)

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    return face_detector, landmark_predictor


detector = None
predictor = None


def ensure_predictor():
    global detector, predictor
    if detector is None or predictor is None:
        detector, predictor = prepare_predictor()
    return detector, predictor


# ------------------------------------------------------------
# 5. FACE ALIGNMENT (ROLL CORRECTION)
# ------------------------------------------------------------

def align_face_and_landmarks(image, landmarks):
    """Rotate image and landmarks so the eye-line is horizontal."""
    left_eye_center = landmarks[36:42].mean(axis=0)
    right_eye_center = landmarks[42:48].mean(axis=0)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        float((left_eye_center[0] + right_eye_center[0]) / 2.0),
        float((left_eye_center[1] + right_eye_center[1]) / 2.0),
    )

    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    h, w = image.shape
    aligned_image = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    pts_homo = np.hstack([landmarks, ones])
    aligned_landmarks = pts_homo @ rot_mat.T

    return aligned_image, aligned_landmarks.astype(np.float32)


# ------------------------------------------------------------
# 6. GEOMETRIC, HOG, AND SPATIAL LBP FEATURES
# ------------------------------------------------------------

def extract_geometric_features(landmarks):
    """Scale- and translation-invariant distances and relative positions."""
    nose_tip = landmarks[30]
    left_eye_center = landmarks[36:42].mean(axis=0)
    right_eye_center = landmarks[42:48].mean(axis=0)
    iod = np.linalg.norm(right_eye_center - left_eye_center) + 1e-6

    norm_pts = (landmarks - nose_tip) / iod

    # Focus on expressive facial regions: eyebrows (17-26), nose (27-35), eyes (36-47), mouth (48-67)
    key_indices = list(range(17, 68))
    selected_pts = norm_pts[key_indices]
    distances_to_nose = np.linalg.norm(selected_pts, axis=1)

    return np.concatenate([selected_pts.ravel(), distances_to_nose]).astype(np.float32)


def extract_hog_region(gray_face, points, target_size=(48, 32), pad=4):
    """Extract HOG from a facial region defined by landmark points."""
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

    return hog(
        roi,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def extract_spatial_lbp(gray_face, grid_size=(6, 6)):
    """Extract uniform LBP histograms over a spatial grid to preserve wrinkle locations."""
    lbp = local_binary_pattern(gray_face, P=8, R=1, method="uniform")
    h, w = lbp.shape
    gh, gw = h // grid_size[0], w // grid_size[1]

    histograms = []
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            patch = lbp[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
            hist, _ = np.histogram(
                patch.ravel(),
                bins=np.arange(0, 11),
                range=(0, 10),
            )
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            histograms.append(hist)

    return np.concatenate(histograms)


# ------------------------------------------------------------
# 7. COMBINED FEATURE EXTRACTION PIPELINE PER IMAGE
# ------------------------------------------------------------

def extract_features_from_image(image):
    if image is None:
        return None

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    ensure_predictor()
    detector_local, predictor_local = ensure_predictor()

    # Upscale 48x48 images 4x to help dlib detect faces and landmarks accurately
    scale = 4
    large_image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    faces = detector_local(large_image, 1)

    if len(faces) > 0:
        face = max(faces, key=lambda r: r.width() * r.height())
    else:
        h, w = large_image.shape
        face = dlib.rectangle(0, 0, w - 1, h - 1)

    shape = predictor_local(large_image, face)

    landmarks = np.array(
        [[shape.part(i).x / scale, shape.part(i).y / scale] for i in range(68)],
        dtype=np.float32,
    )

    # 1. Align face roll based on eye centers
    aligned_image, aligned_landmarks = align_face_and_landmarks(image, landmarks)

    # Crop tightly around the aligned 68 landmarks
    min_xy = aligned_landmarks.min(axis=0)
    max_xy = aligned_landmarks.max(axis=0)
    margin_x = (max_xy[0] - min_xy[0]) * 0.08
    margin_y = (max_xy[1] - min_xy[1]) * 0.12

    x1 = max(0, int(np.floor(min_xy[0] - margin_x)))
    y1 = max(0, int(np.floor(min_xy[1] - margin_y)))
    x2 = min(aligned_image.shape[1], int(np.ceil(max_xy[0] + margin_x)) + 1)
    y2 = min(aligned_image.shape[0], int(np.ceil(max_xy[1] + margin_y)) + 1)

    if x2 <= x1 or y2 <= y1:
        return None

    face_image = aligned_image[y1:y2, x1:x2]
    if face_image.size == 0:
        return None

    crop_landmarks = aligned_landmarks.copy()
    crop_landmarks[:, 0] -= x1
    crop_landmarks[:, 1] -= y1

    old_h, old_w = face_image.shape
    face_size = 96
    face_image = cv2.resize(
        face_image,
        (face_size, face_size),
        interpolation=cv2.INTER_CUBIC,
    )

    crop_landmarks[:, 0] *= face_size / max(old_w, 1)
    crop_landmarks[:, 1] *= face_size / max(old_h, 1)
    crop_landmarks = np.clip(crop_landmarks, 0, face_size - 1)

    landmark_features = extract_geometric_features(crop_landmarks)

    pts_int = crop_landmarks.astype(np.int32)
    left_eye_brow_pts = np.vstack([pts_int[17:22], pts_int[36:42]])
    right_eye_brow_pts = np.vstack([pts_int[22:27], pts_int[42:48]])
    nose_pts = pts_int[27:36]
    mouth_pts = pts_int[48:68]

    left_hog = extract_hog_region(face_image, left_eye_brow_pts, target_size=(48, 32), pad=4)
    right_hog = extract_hog_region(face_image, right_eye_brow_pts, target_size=(48, 32), pad=4)
    nose_hog = extract_hog_region(face_image, nose_pts, target_size=(32, 48), pad=5)
    mouth_hog = extract_hog_region(face_image, mouth_pts, target_size=(64, 32), pad=4)

    if any(feat is None for feat in (left_hog, right_hog, nose_hog, mouth_hog)):
        return None

    hog_features = np.concatenate([left_hog, right_hog, nose_hog, mouth_hog])
    lbp_features = extract_spatial_lbp(face_image, grid_size=(6, 6))

    return np.concatenate(
        [landmark_features, hog_features, lbp_features]
    ).astype(np.float32)


def extract_all_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return extract_features_from_image(image)


# ------------------------------------------------------------
# 8. HELPER: EXTRACT SUBJECT ID FROM PATH
# ------------------------------------------------------------

def get_subject_id(filepath):
    p_obj = Path(filepath)
    filename = p_obj.name

    if filename.startswith("S") and len(filename) >= 4 and filename[1:4].isdigit():
        return filename[:4]

    for part in p_obj.parts:
        if part.startswith("S") and part[1:].isdigit():
            return part

    return p_obj.stem.split("_")[0]


def build_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full")),
            (
                "classifier",
                LogisticRegression(
                    C=LOGREG_C,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first with python train.py --train")
    return joblib.load(model_path)


def predict_image_path(image_path, model=None):
    if model is None:
        model = load_model()

    features = extract_all_features(str(image_path))
    if features is None or not np.isfinite(features).all():
        raise ValueError(f"Unable to extract valid features from: {image_path}")

    return predict_vector(features, model)


def predict_image_path_from_array(image, model=None):
    if model is None:
        model = load_model()

    features = extract_features_from_image(image)
    if features is None or not np.isfinite(features).all():
        raise ValueError("Unable to extract valid features from the uploaded image.")

    return predict_vector(features, model)


def predict_vector(features, model):
    feature_row = features.reshape(1, -1)
    probabilities = model.predict_proba(feature_row)[0]
    class_names = list(model.classes_)
    scores = {label: float(probability) for label, probability in zip(class_names, probabilities)}
    for label in EMOTIONS:
        scores.setdefault(label, 0.0)

    dominant = max(scores, key=scores.get)
    confidence = float(scores.get(dominant, 0.0))
    return {"dominant": dominant, "confidence": confidence, "scores": scores}


# ------------------------------------------------------------
# 9. MAIN PIPELINE: EXTRACT, 5-FOLD SUBJECT-INDEPENDENT CV, SAVE
# ------------------------------------------------------------

def main():
    image_paths, labels = load_dataset(IMAGE_ROOT)

    feature_rows = []
    successful_paths = []
    successful_labels = []
    failed = 0

    print("\n========== FEATURE EXTRACTION ==========")

    for image_path, label in tqdm(
        zip(image_paths, labels),
        total=len(image_paths),
    ):
        try:
            features = extract_all_features(image_path)

            if features is None or not np.isfinite(features).all():
                failed += 1
                continue

            feature_rows.append(features)
            successful_paths.append(image_path)
            successful_labels.append(label)

        except Exception as error:
            failed += 1
            print(f"\nSkipped {image_path}: {error}")

    if len(feature_rows) < 2:
        raise ValueError(
            "Not enough images were processed successfully. "
            "Check the images and dlib predictor file."
        )

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.asarray(successful_labels)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Images found:", len(image_paths))
    print("Images processed:", len(X))
    print("Images skipped:", failed)
    print("Features per image:", X.shape[1])

    feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
    raw_df = pd.DataFrame(X, columns=feature_columns)
    raw_df.insert(0, "label", y)
    raw_df.insert(0, "image_path", successful_paths)

    raw_csv_path = OUTPUT_DIR / "ck48_features.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print("Raw feature CSV saved:", raw_csv_path)

    # --------------------------------------------------------
    # 10. 5-FOLD SUBJECT-INDEPENDENT CROSS-VALIDATION
    # --------------------------------------------------------

    subject_ids = np.asarray([get_subject_id(p) for p in successful_paths])
    print(f"Unique subjects detected: {len(np.unique(subject_ids))}")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_predictions = np.empty_like(y)
    fold_accuracies = []
    best_accuracy = -1.0
    best_model = None
    best_fold = 1

    print(f"\n========== {N_SPLITS}-FOLD SUBJECT-INDEPENDENT CV ==========")

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=subject_ids), start=1):
        overlap = set(subject_ids[train_idx]).intersection(set(subject_ids[test_idx]))
        assert len(overlap) == 0, f"Subject leakage detected in fold {fold}: {overlap}"

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = build_pipeline()
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        oof_predictions[test_idx] = y_pred
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        n_comp = fold_model.named_steps["pca"].n_components_
        print(f"Fold {fold}/{N_SPLITS} | Test samples: {len(test_idx):3d} | PCA dims: {n_comp:3d} | Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = fold_model
            best_fold = fold

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    overall_oof_acc = accuracy_score(y, oof_predictions)

    class_labels = best_model.named_steps["classifier"].classes_
    overall_report = classification_report(y, oof_predictions, zero_division=0)
    overall_matrix = confusion_matrix(y, oof_predictions, labels=class_labels)

    print("\n========== OVERALL 5-FOLD CV RESULTS ==========")
    print(f"Mean Fold Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Overall OOF Accuracy: {overall_oof_acc:.4f} ({np.sum(y == oof_predictions)}/{len(y)})")
    print(f"Best Fold Model    : Fold {best_fold} ({best_accuracy:.4f})")
    print("\nOverall Classification Report (All Subjects Out-of-Fold):\n", overall_report)

    # --------------------------------------------------------
    # 11. SAVE BEST MODEL, FULL-DATASET MODEL, AND REPORTS
    # --------------------------------------------------------

    # Fit a final production model on 100% of the dataset for future inference
    final_full_model = build_pipeline()
    final_full_model.fit(X, y)

    model_path = OUTPUT_DIR / "ck48_landmarks_hog_lbp_pca_logreg.joblib"
    report_path = OUTPUT_DIR / "ck48_classification_report.txt"
    matrix_path = OUTPUT_DIR / "ck48_confusion_matrix.csv"
    pca_csv_path = OUTPUT_DIR / "ck48_features_pca.csv"

    joblib.dump(final_full_model, model_path)
    joblib.dump(final_full_model, MODEL_PATH)
    report_path.write_text(
        f"5-Fold Subject-Independent Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"Overall Out-of-Fold Accuracy: {overall_oof_acc:.4f}\n\n"
        + overall_report,
        encoding="utf-8",
    )
    pd.DataFrame(
        overall_matrix,
        index=class_labels,
        columns=class_labels,
    ).to_csv(matrix_path)

    X_all_scaled = final_full_model.named_steps["scaler"].transform(X)
    X_all_pca = final_full_model.named_steps["pca"].transform(X_all_scaled)

    pca_columns = [f"pca_component_{i + 1}" for i in range(X_all_pca.shape[1])]
    pca_df = pd.DataFrame(X_all_pca, columns=pca_columns)
    pca_df.insert(0, "label", y)
    pca_df.insert(0, "oof_prediction", oof_predictions)
    pca_df.insert(0, "subject_id", subject_ids)
    pca_df.insert(0, "image_path", successful_paths)
    pca_df.to_csv(pca_csv_path, index=False)

    print("\nSaved files:")
    print("Production Model (100% data):", model_path)
    print("Integrated app model:", MODEL_PATH)
    print("Report:", report_path)
    print("Confusion matrix CSV:", matrix_path)
    print("PCA & OOF Predictions CSV:", pca_csv_path)

    # --------------------------------------------------------
    # 12. PLOT OVERALL OUT-OF-FOLD CONFUSION MATRIX
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=overall_matrix,
        display_labels=class_labels,
    )
    display.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title(f"CK+48 5-Fold Subject-Independent CV (Acc: {overall_oof_acc:.2%})")
    plt.tight_layout()

    confusion_plot_path = OUTPUT_DIR / "ck48_confusion_matrix.png"
    plt.savefig(confusion_plot_path, dpi=200)
    plt.show()

    print("Confusion matrix plot:", confusion_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train, predict, or serve the emotion recognition app.")
    parser.add_argument("--train", action="store_true", help="Retrain the model from the CK+48 dataset.")
    parser.add_argument("--predict", type=str, metavar="IMAGE_PATH", help="Predict a single image file.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Model path to use for prediction.")
    parser.add_argument("--app", action="store_true", help="Start the Flask webcam web app from this single file.")
    args = parser.parse_args()

    if args.app:
        app.run(host="127.0.0.1", port=5000, debug=False)
    elif args.train:
        main()
    elif args.predict:
        model = load_model(Path(args.model))
        result = predict_image_path(Path(args.predict), model=model)
        print(result)
    else:
        if MODEL_PATH.exists():
            app.run(host="127.0.0.1", port=5000, debug=False)
        else:
            main()