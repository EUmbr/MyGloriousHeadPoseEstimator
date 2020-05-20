import cv2
import numpy as np
import math


class HeadPoseEstimator:

    def __init__(self, image_size):
        self.size = image_size

        self.points_3d = self._get_full_model_points()

        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        self.dist_coeefs = np.zeros((4, 1))

        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

    def _get_full_model_points(self, filename='models/model.txt'):
        """CHANGE IT!!!!!!!!!!!!!"""

        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.points_3d, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def get_vectors(self, rotation_vector, translation_vector):
        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])

        imgpts, jac = cv2.projectPoints(
            axis, rotation_vector, translation_vector, self.camera_matrix, self.dist_coeefs)
        modelpts, jac2 = cv2.projectPoints(
            self.points_3d, rotation_vector, translation_vector, self.camera_matrix, self.dist_coeefs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

        return (str(int(roll)), str(int(pitch)), str(int(yaw)))

    def draw_face_direction(self, image, nose, rotation_vector, translation_vector, color=(255, 0, 0), line_width=2):
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 200.0)]), rotation_vector,
                                                         translation_vector, self.camera_matrix, self.dist_coeefs)

        p1 = (int(nose[0]), int(nose[1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(image, p1, p2, color, line_width)

    def draw_all_landmarks(self, image, landmarks):
        landmarks = landmarks.astype(int)
        for landmark in landmarks:
            cv2.circle(image, tuple(landmark), 2, (71, 99, 255), 1)

    def draw_mask(self, image, landmarks):
        landmarks = landmarks.astype(int)
        for i in range(16):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[0]), tuple(
            landmarks[17]-[0, 35]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[16]), tuple(
            landmarks[26]-[0, 35]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[17]-[0, 35]),
                 tuple(landmarks[18]-[0, 35]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[26]-[0, 35]),
                 tuple(landmarks[25]-[0, 35]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[18]-[0, 35]),
                 tuple(landmarks[19]-[0, 35]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[25]-[0, 35]),
                 tuple(landmarks[24]-[0, 35]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[19]-[0, 35]),
                 tuple(landmarks[20]-[0, 40]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[24]-[0, 35]),
                 tuple(landmarks[23]-[0, 40]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[20]-[0, 40]),
                 tuple(landmarks[21]-[0, 47]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[23]-[0, 40]),
                 tuple(landmarks[22]-[0, 47]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[21]-[0, 47]),
                 tuple(landmarks[22]-[0, 47]), (221, 226, 179), 1)

        for i in range(17, 21):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)

        for i in range(22, 26):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)

        for i in range(36, 41):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[41]), tuple(
            landmarks[36]), (221, 226, 179), 1)

        for i in range(42, 47):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[47]), tuple(
            landmarks[42]), (221, 226, 179), 1)

        for i in range(27, 30):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)

        for i in range(31, 35):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)

        for i in range(48, 59):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[59]), tuple(
            landmarks[48]), (221, 226, 179), 1)

        for i in range(60, 67):
            cv2.line(image, tuple(landmarks[i]), tuple(
                landmarks[i+1]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[67]), tuple(
            landmarks[60]), (221, 226, 179), 1)

        lm = [1, 15, 8, 1]
        for i in range(3):
            cv2.circle(image, tuple(landmarks[lm[i]]), 3, (0, 0, 255), 1)
            cv2.line(image, tuple(landmarks[lm[i]]), tuple(
                landmarks[lm[i+1]]), (0, 0, 150), 1)

        cv2.line(image, tuple(landmarks[31]), tuple(
            landmarks[48]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[35]), tuple(
            landmarks[54]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[35]), tuple(
            landmarks[15]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[45]), tuple(
            landmarks[15]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[31]), tuple(
            landmarks[1]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[36]), tuple(
            landmarks[1]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[39]), tuple(
            landmarks[31]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[35]), tuple(
            landmarks[42]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[33]), tuple(
            landmarks[50]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[33]), tuple(
            landmarks[52]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[5]), tuple(
            landmarks[48]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[11]), tuple(
            landmarks[54]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[8]), tuple(
            landmarks[57]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[8]), tuple(
            landmarks[48]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[8]), tuple(
            landmarks[54]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[54]), tuple(
            landmarks[14]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[35]), tuple(
            landmarks[14]), (221, 226, 179), 1)

        cv2.line(image, tuple(landmarks[31]), tuple(
            landmarks[2]), (221, 226, 179), 1)
        cv2.line(image, tuple(landmarks[48]), tuple(
            landmarks[2]), (221, 226, 179), 1)

    def draw_vectors(self, image):
        label_roll = "roll: %.4f" % self.roll
        label_size, base_line = cv2.getTextSize(
            label_roll, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(image, label_roll, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        label_pitch = "pitch: %.4f" % self.pitch
        label_size, base_line = cv2.getTextSize(
            label_pitch, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(image, label_pitch, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        label_yaw = "yaw: %.4f" % self.yaw
        label_size, base_line = cv2.getTextSize(
            label_yaw, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(image, label_yaw, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
