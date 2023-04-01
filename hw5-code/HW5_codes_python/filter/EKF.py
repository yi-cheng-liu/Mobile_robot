import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun    # motion model
        self.hfun = system.hfun    # measurement model
        self.Gfun = init.Gfun      # Jocabian of motion model
        self.Vfun = init.Vfun      # Jocabian of motion model
        self.Hfun = init.Hfun      # Jocabian of measurement model
        self.M = system.M          # motion noise covariance
        self.Q = system.Q          # measurement noise covariance

        self.gfun = system.gfun   # motion model
        self.hfun = system.hfun   # measurement model
        self.Gfun = init.Gfun     # Jocabian of motion model
        self.Vfun = init.Vfun     # Jocabian of motion model
        self.Hfun = init.Hfun     # Jocabian of measurement model
        self.M = system.M         # motion noise covariance
        self.Q = system.Q         # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u):

        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        # Prediction step

        # 1. State prediction
        X_pred = self.gfun(X, u)

        # 2. Covariance prediction with mean and input
        G_t = self.Gfun(X, u)
        V_t = self.Vfun(X, u)

        P_pred = G_t @ P @ G_t.T + V_t @ self.M(u) @ V_t.T

        # 2. Covariance prediction
        G_t = self.Gfun(X, u)
        V_t = self.Vfun(X, u)
        P_pred = G_t @ P @ G_t.T + V_t @ self.M(u) @V_t.T


        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = self.state_.getState()         # mean
        P_predict = self.state_.getCovariance()    # covariance
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        
        # Two landmarks
        landmark_x1, landmark_y1 = landmark1.getPosition()[0], landmark1.getPosition()[1]
        landmark_x2, landmark_y2 = landmark2.getPosition()[0], landmark2.getPosition()[1]

        # Predicted mean
        z_hat1 = self.hfun(landmark_x1, landmark_y1, X_predict)
        z_hat2 = self.hfun(landmark_x2, landmark_y2, X_predict)

        # Jacobian
        H1 = self.Hfun(landmark_x1, landmark_y1, X_predict, z_hat1)
        H2 = self.Hfun(landmark_x2, landmark_y2, X_predict, z_hat2)
        H = np.vstack((H1, H2))      # (4,3)

        # innovation - (measurement mean - predicted mean)
        innovation = [wrap2Pi(z[0] - z_hat1[0]), z[1] - z_hat1[1],
                      wrap2Pi(z[3] - z_hat2[0]), z[4] - z_hat2[1]]
        
        # Innovation covariance
        QQ = block_diag(self.Q, self.Q)    # two measurement noise covariance
        S = H @ P_predict @ H.T + QQ       # (4,3)

        # Kalman gain
        K = P_predict @ H.T @ np.linalg.inv(S) # (3,4)

        # Correct mean
        X = X_predict + K @ innovation   # (3, )
        X[2] = wrap2Pi(X[2])

        # Corrected covariance
        D = np.eye(len(X)) - K @ H
        P = D @ P_predict @ D.T + K @ QQ @ K.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state