import dlib
import cv2
import numpy as np


class FaceDetector:
    """Detect face on image"""

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt',
            'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.detector = dlib.get_frontal_face_detector()
        self.detected_result = None

    def detecte_face(self, image):
        faces, scores, idx = self.detector.run(image)
        self.detected_result = faces
        return faces

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
            face[0] -=diff/2
            face[2] += diff/2
            face = face.astype('int')
            
            face = dlib.rectangle(face[0], face[1], face[2], face[3])

            return success, face

        else:
            return False, None

    def draw_face(self, image, face, color=(170, 205, 102)):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)


class LandmarkDetector:
    """Facial landmark detector"""

    def __init__(self):
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

    def draw_landmarks(self, image, landmarks, color=(71, 99, 255)):
        for landmark in landmarks:
            x = landmarks.part(landmark).x
            y = landmarks.part(landmark).y

            cv2.circle(image, (x, y), 4, color, -1)
