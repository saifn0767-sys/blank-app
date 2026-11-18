import dlib

_DETECTOR = dlib.get_frontal_face_detector()

def detect_largest_face(gray, upsample=1):
    faces = _DETECTOR(gray, upsample)
    if not faces:
        return None
    return max(faces, key=lambda r: r.width() * r.height())
