import cv2
import numpy as np


class FaceDetector:
    """Detect face on image"""

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        self.detected_result = None

    def detect_with_dnn(self, image, treshold=0.5):
        (h, w) = image.shape[:2]

        # preprocessing input image
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)

        # apply face detection
        detections = self.net.forward()

        conf = detections[0, 0, 0, 2]
        if conf > treshold:
            success = True
            face = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            diff = (face[3]-face[1]) - (face[2] - face[0])
            face[0] -= diff/2
            face[2] += diff/2
            face[3] -= face[1]
            face[2] -= face[0]
            face = face.astype('int')

            return success, np.array([face])

        else:
            return False, None

    def draw_face(self, image, face, color=(170, 205, 102)):
        x1 = face[0]
        y1 = face[1]
        x2 = face[0] + face[2]
        y2 = face[1] + face[3]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)


class LandmarkDetector:
    """Facial landmark detector"""

    def __init__(self):
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel("models/lbfmodel.yaml")

        self.landmarks = None

    def detect(self, image, face):
        landmarks = np.zeros((68, 2))
        _, self.landmarks = self.landmark_detector.fit(image, face)

        for n in range(0, 68):
            landmarks[n, 0] = self.landmarks[0][0][n][0]
            landmarks[n, 1] = self.landmarks[0][0][n][1]

        return landmarks

    def draw_landmarks(self, image, landmarks, color=(71, 99, 255)):
        for landmark in landmarks:
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 4, color, -1)