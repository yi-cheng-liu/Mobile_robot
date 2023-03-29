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

        # 2. Covariance prediction
        G_t = self.Gfun(X, u)
        V_t = self.Vfun(X, u)
        P_pred = np.dot(np.dot(G_t, P), G_t.T) + np.dot(np.dot(V_t, self.M), V_t.T)


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
        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        # Correction step

        # Measurement prediction
        z_pred = self.hfun(X_predict, landmark1.getPosition(), landmark2.getPosition())

        # Measurement residual
        innovation = z - z_pred
        innovation[1] = wrap2Pi(innovation[1])  # Wrap angle difference to [-pi, pi]
        innovation[4] = wrap2Pi(innovation[4])  # Wrap angle difference to [-pi, pi]

        # Compute the Jacobian H_t
        H_t = self.Hfun(X_predict, landmark1.getPosition(), landmark2.getPosition())

        # Kalman gain
        S = np.dot(np.dot(H_t, P_predict), H_t.T) + self.Q
        K = np.dot(np.dot(P_predict, H_t.T), np.linalg.inv(S))

        # State update
        X = X_predict + np.dot(K, innovation)

        # Covariance update
        P = np.dot((np.eye(3) - np.dot(K, H_t)), P_predict)

        
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