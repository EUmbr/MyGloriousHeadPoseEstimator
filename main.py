import cv2

from detector import LandmarkDetector, FaceDetector
from headpose_estimator import HeadPoseEstimator

cap = cv2.VideoCapture(0)
_, start_frame = cap.read()

height, width = start_frame.shape[:2]
pose_estimator = HeadPoseEstimator(image_size=(height, width))
fd = FaceDetector()
ld = LandmarkDetector()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    success, dnn_face = fd.detect_with_dnn(frame, 0.5)

    if success:
        fd.draw_face(frame, dnn_face, (0, 0, 255))
        landmarks = ld.detect(gray, dnn_face)
        nose = landmarks[30]

        head_pose = pose_estimator.solve_pose_by_68_points(landmarks)

        pose_estimator.draw_face_direction(
            frame, nose, head_pose[0], head_pose[1])
        pose_estimator.draw_all_landmarks(frame, landmarks)
        pose_estimator.draw_mask(frame, landmarks)

        roll, pitch, yaw = pose_estimator.get_vectors(
            head_pose[0], head_pose[1])
        pose_estimator.draw_vectors(frame)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
