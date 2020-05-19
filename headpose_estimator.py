import cv2
import numpy as np


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

    def _get_full_model_points(self, filename='model.txt'):
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

    def draw_face_direction(self, image, nose, rotation_vector, translation_vector, color=(255, 0, 0), line_width=2):
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 200.0)]), rotation_vector,
                                                         translation_vector, self.camera_matrix, self.dist_coeefs)

        p1 = (int(nose[0]), int(nose[1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(image, p1, p2, color, line_width)

    def draw_all_landmarks(self, image, marks):
        marks = marks.astype(int)
        for mark in marks:
            cv2.circle(image, tuple(mark), 2, (71, 99, 255), 1)
