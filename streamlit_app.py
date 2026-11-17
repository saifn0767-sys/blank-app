"""
Streamlit emotion dashboard: camera capture + real-time -> CK+48 model -> CSV log for Tableau.

Place next to ck48_features.py (or copy it here). Run once ck48_features.py has produced the .joblib model:
    pip install -r requirements_streamlit.txt
    streamlit run streamlit_app.py
"""
import csv
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
LOG_PATH = HERE / "predictions_log.csv"
LOCK = threading.Lock()

st.set_page_config(page_title="Emotion Dashboard", layout="wide")


@st.cache_resource
def load_everything():
    import ck48_features as ck  # loads dlib detector/predictor once
    name = "ck48_landmarks_hog_lbp_pca_logreg.joblib"
    local = HERE / name                      # cloud: put the model file in the repo root
    model = joblib.load(local if local.exists() else ck.OUTPUT_DIR / name)
    return ck, model


ck, model = load_everything()
CLASSES = list(model.named_steps["classifier"].classes_)
HEADER = ["timestamp", "date", "hour", "emotion", "confidence"] + [f"p_{c}" for c in CLASSES]
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADER)


def predict_frame(frame_bgr):
    """Returns (emotion, confidence, probs, (x, y, w, h) in pixels) or None if no face."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = ck.detector(gray, 1)
    if not faces:
        return None
    f = max(faces, key=lambda r: r.width() * r.height())
    m = int(0.1 * f.width())
    x1, y1 = max(0, f.left() - m), max(0, f.top() - m)
    x2, y2 = min(gray.shape[1], f.right() + m), min(gray.shape[0], f.bottom() + m)
    face = gray[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "face.png"
        cv2.imwrite(str(p), face)
        feats = ck.extract_all_features(p)
    if feats is None or not np.isfinite(feats).all():
        return None
    probs = model.predict_proba(feats.reshape(1, -1))[0]
    i = int(np.argmax(probs))
    return CLASSES[i], float(probs[i]), dict(zip(CLASSES, map(float, probs))), (x1, y1, x2 - x1, y2 - y1)


def log_prediction(emotion, conf, probs):
    now = datetime.now()
    row = [now.isoformat(timespec="seconds"), now.date().isoformat(), now.hour, emotion, round(conf, 4)]
    row += [round(probs[c], 4) for c in CLASSES]
    with LOCK, open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest = None          # (emotion, conf, probs)
        self.last_pred = 0.0
        self.last_log = 0.0
        self.box = None
        self.label = ""

    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        now = time.time()
        if now - self.last_pred > 0.3:       # throttle: predict ~3x per second
            self.last_pred = now
            res = predict_frame(img)
            if res:
                emo, conf, probs, self.box = res
                self.latest = (emo, conf, probs)
                self.label = f"{emo} {conf:.0%}"
                if now - self.last_log > 2:  # log every 2 s
                    self.last_log = now
                    log_prediction(emo, conf, probs)
            else:
                self.box, self.label, self.latest = None, "", None
        if self.box:
            x, y, w, h = self.box
            cv2.rectangle(img, (x, y), (x + w, y + h), (180, 212, 45), 2)
            cv2.putText(img, self.label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 212, 45), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def show_probs(container, latest):
    if latest is None:
        container.info("No face found. Face the camera in good light.")
        return
    emo, conf, probs = latest
    with container.container():
        st.subheader(f"{emo} ({conf:.0%})")
        for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
            st.progress(v, text=f"{k}  {v:.0%}")


st.title("Emotion dashboard")
tab_live, tab_cap, tab_tab = st.tabs(["Real-time", "Capture", "Tableau analysis"])

with tab_live:
    left, right = st.columns([3, 2])
    with left:
        ctx = webrtc_streamer(
            key="live",
            video_processor_factory=LiveProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
    holder = right.empty()
    if ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                show_probs(holder, ctx.video_processor.latest)
            time.sleep(0.4)
    else:
        holder.caption("Press Start to begin. Every 2 seconds one prediction is saved to the log.")

with tab_cap:
    photo = st.camera_input("Take a photo")
    if photo is not None:
        frame = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        res = predict_frame(frame)
        if res is None:
            st.error("No face found. Retake the photo facing the camera.")
        else:
            emo, conf, probs, _ = res
            log_prediction(emo, conf, probs)
            show_probs(st.container(), (emo, conf, probs))
            st.success("Saved to log.")

with tab_tab:
    df = pd.read_csv(LOG_PATH)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Logged predictions", len(df))
        st.download_button("Download CSV for Tableau", df.to_csv(index=False), "predictions_log.csv", "text/csv")
        st.dataframe(df.tail(10).iloc[::-1][["timestamp", "emotion", "confidence"]], hide_index=True)
        st.caption("Connect Tableau to predictions_log.csv, publish the dashboard, then paste its URL.")
    with c2:
        url = st.text_input("Published Tableau dashboard URL")
        if url:
            components.iframe(url + ("&" if "?" in url else "?") + ":embed=y&:showVizHome=no", height=600)
