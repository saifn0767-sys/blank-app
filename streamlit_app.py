from pathlib import Path
import time
import uuid
import threading

import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from vision.feature_extraction import extract_all_features, load_predictor, EXPECTED_FEATURES
from database.db import initialize_db, log_prediction, read_logs, is_cloud, EMOTIONS
from dashboard.visualizations import show_dashboard

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'models' / 'ck48_final_model.joblib'
PREDICTOR_FILE = ROOT / 'shape_predictor_68_face_landmarks.dat'



st.set_page_config(page_title='Facial Emotion System', page_icon='🙂', layout='wide')


@st.cache_resource
def load_assets():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f'Model missing: {MODEL_FILE}')
    if not PREDICTOR_FILE.exists():
        raise FileNotFoundError(f'Dlib predictor missing: {PREDICTOR_FILE}')
    model = joblib.load(MODEL_FILE)
    n_in = getattr(model, 'n_features_in_', None)
    if n_in is not None and n_in != EXPECTED_FEATURES:
        raise ValueError(f'Model expects {n_in} features but the extractor produces {EXPECTED_FEATURES}. Use the model trained by the CK+ script.')
    return model, load_predictor(PREDICTOR_FILE)


initialize_db()


# Shared, thread-safe data exchanged between WebRTC frame callback and Streamlit UI.
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {'label': None, 'probabilities': {}, 'confidence': None, 'error': None, 'face_found': False}
        self.session_id = uuid.uuid4().hex[:10]
        self.last_logged = 0.0
        self.prediction_count = 0


if 'shared_state' not in st.session_state:
    st.session_state.shared_state = SharedState()
state = st.session_state.shared_state


try:
    model, predictor = load_assets()
except Exception as exc:
    st.error(f'Cannot load model assets: {exc}')
    st.info('Place models/ck48_final_model.joblib and shape_predictor_68_face_landmarks.dat in the project root as described in README.')
    st.stop()


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_log_time = 0.0

        # Assets are loaded from Streamlit cached resource in the main thread.
        self.model = model
        self.predictor = predictor
        self.shared = state


    def recv(self, frame):
        image = frame.to_ndarray(format='bgr24')
        display = image.copy()
        try:
            features, face, points = extract_all_features(image, self.predictor)
            if features is None:
                with self.shared.lock:
                    self.shared.latest = {'label': None, 'probabilities': {}, 'confidence': None, 'error': None, 'face_found': False}
                cv2.putText(display, 'No face detected', (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)
            else:
                predicted = self.model.predict(features)[0]
                raw_prob = self.model.predict_proba(features)[0]
                classes = [str(x) for x in self.model.classes_]
                probs = {c: float(p) for c, p in zip(classes, raw_prob)}
                label = str(predicted)
                confidence = float(probs.get(label, np.max(raw_prob)))
                # Draw landmarks and label on video frame.
                for x, y in points.astype(int):
                    if 0 <= x < display.shape[1] and 0 <= y < display.shape[0]:
                        cv2.circle(display, (x, y), 1, (50, 220, 80), -1)
                cv2.putText(display, f'{label.upper()}  {confidence:.1%}', (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 220, 90), 2, cv2.LINE_AA)
                with self.shared.lock:
                    self.shared.latest = {'label': label, 'probabilities': probs, 'confidence': confidence, 'error': None, 'face_found': True}
                    self.shared.prediction_count += 1
                now = time.time()
                # Limit database writes to once every 2 seconds per stream.
                if now - self.last_log_time >= 2.0:
                    log_prediction(self.shared.session_id, label, probs)
                    self.last_log_time = now
        except Exception as exc:
            with self.shared.lock:
                self.shared.latest = {'label': None, 'probabilities': {}, 'confidence': None, 'error': str(exc), 'face_found': False}
            cv2.putText(display, 'Processing error', (18, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return frame.from_ndarray(display, format='bgr24')


st.title('Facial Emotion Recognition System')
st.caption('CK+ facial-expression classifier • Handcrafted features • Multinomial Logistic Regression • No deep learning')
page = st.sidebar.radio('Navigation', ['Live Recognition', 'Analytics Dashboard', 'Stored Data'])
st.sidebar.caption('Storage: ' + ('cloud SQL database' if is_cloud() else 'local SQLite (set DATABASE_URL for cloud)'))

if page == 'Live Recognition':
    st.markdown('### Live camera')
    col_video, col_result = st.columns([1.65, 1])
    with col_video:
        webrtc_streamer(
            key='emotion-live-camera',
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={'video': True, 'audio': False},
            async_processing=True,
            rtc_configuration={'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]},
        )
    with col_result:
        st.markdown('### Current prediction')
        st_autorefresh(interval=1200, key='live_metrics_refresh')
        with state.lock:
            latest = dict(state.latest)
            probs = dict(latest.get('probabilities') or {})
            count = state.prediction_count
        if latest.get('error'):
            st.error(latest['error'])
        if latest.get('label'):
            st.metric('Predicted expression', latest['label'].upper())
            st.metric('Predicted class probability', f"{latest['confidence']:.1%}")
            st.metric('Frames processed', count)
            if probs:
                st.bar_chart({k: v for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)})
        else:
            st.info('Start the camera and position your face in view.')
    st.caption('Predicted facial-expression labels are not direct measurements of a person’s internal emotional state. Images are not saved by this app.')

elif page == 'Analytics Dashboard':
    st.header('Analytics Dashboard')
    st_autorefresh(interval=5000, key='analytics_refresh')
    show_dashboard(read_logs())

else:
    st.header('Stored data (SQL table: emotion_log)')
    st.dataframe(read_logs(), use_container_width=True)
