import dlib
import cv2
import numpy as np


class FaceDetector:
    """Detect face on image"""

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.detected_result = None

    def get_detected_face(self, image):
        faces, scores, idx = self.detector.run(image)
        self.detected_result = faces
        return faces

    def draw_faces(self, image, face):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), (170, 205, 102), 3)


class LandmarkDetector:
    """Facial landmark detector"""

    def __init__(self):
        self.detector = FaceDetector()
        self.landmark_detector = dlib.shape_predictor(
            "models/shape_predictor_68_face_landmarks.dat")

        self.landmarks = None

    def detect(self, image, face):
        landmarks = np.zeros((68, 2))
        self.landmarks = self.landmark_detector(image, face)

        for n in range(0, 68):
            x = self.landmarks.part(n).x
            landmarks[n, 0] = x
            y = self.landmarks.part(n).y
            landmarks[n, 1] = y

        return landmarks

    def draw_landmarks(self, image, landmarks):
        for landmark in landmarks:
            x = landmarks.part(landmark).x
            y = landmarks.part(landmark).y

            cv2.circle(image, (x, y), 4, (71, 99, 255), -1)
