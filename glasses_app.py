import streamlit as st
import cv2
import av
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# إعداد MediaPipe للكشف عن الوجه
mp_face_mesh = mp.solutions.face_mesh

class GlassesDetector(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # تحليل الوجه
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # استخراج ملامح الوجه
                left_eye = [face_landmarks.landmark[i] for i in range(33, 133)]
                right_eye = [face_landmarks.landmark[i] for i in range(362, 464)]
                nose_bridge = [face_landmarks.landmark[i] for i in [6, 168]]

                # رسم النقاط
                for landmark in left_eye + right_eye + nose_bridge:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # كشف النظارات (بناءً على موقع جسر الأنف والعين)
                nose_top = nose_bridge[0].y
                nose_bottom = nose_bridge[1].y
                eye_level = (left_eye[0].y + right_eye[0].y) / 2

                if eye_level < nose_top < nose_bottom:
                    cv2.putText(image, "نظارة مكتشفة 🕶️", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "لا توجد نظارة", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# إعداد تطبيق Streamlit
st.title("🔍 كشف النظارات عبر الكاميرا")

st.write("افتح الكاميرا وشوف إذا كنت لابس نظارة ولا لأ!")

# تشغيل الكاميرا
webrtc_streamer(key="camera", video_transformer_factory=GlassesDetector)
