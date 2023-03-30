
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)



    def prediction(self, u):
        # prior belief
        X = self.state_.getState()         # mean
        P = self.state_.getCovariance()    # covariance 

        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        # 1. The noise is multiplicative, thus argument the state
        # Adding zero to mean
        X_aug = np.hstack((X, np.zeros(X.shape[0]))).reshape(-1, 1)
        # onstructing convariance matrix of the state and noise convariance
        P_aug = block_diag(P, self.M(u))

        # 2. compute the sigma points, and the weights
        self.sigma_point(X_aug, P_aug, self.kappa_g)

        # 3. prediction mean and covariance
        X_pred = np.zeros((3, 1))
        self.F = np.zeros((3, 2*self.n + 1))   # f(u_k, x_prev, i)
        for i in range(2*self.n + 1):
            self.F[:, i] = self.gfun(self.X[:3, i], u + self.X[3:, i])
            X_pred += self.w[i] * self.F[:, i].reshape(-1, 1)
        P_pred = (self.F - X_pred) @ np.diag(self.w) @ (self.F - X_pred).T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):

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

        landmark_x1, landmark_y1 = landmark1.getPosition()[0], landmark1.getPosition()[1]
        landmark_x2, landmark_y2 = landmark2.getPosition()[0], landmark2.getPosition()[1]

        Z_all = np.zeros((4, 2*self.n + 1))
        z_hat = np.zeros((4, 1))
        S = np.zeros((4, 4)) 
        P_xz = np.zeros((3, 4))

        for i in range(2 * self.n + 1):
            Z1 = self.hfun(landmark_x1, landmark_y1, self.F[:,i])
            Z2 = self.hfun(landmark_x2, landmark_y2, self.F[:,i])
            Z_all[:, i] = np.hstack((Z1, Z2)) 

            # Predicted mean
            z_hat += self.w[i] * Z_all[:, i].reshape(-1, 1)

        # Innovation - (measurement mean - predicted mean)
        innovation = [wrap2Pi(z[0] - z_hat[0]), z[1] - z_hat[1],
                        wrap2Pi(z[3] - z_hat[2]), z[4] - z_hat[3]]
        
        # Innovation covariance
        QQ = block_diag(self.Q, self.Q)    # two measurement noise covariance
        S = ((Z_all - z_hat) @ np.diag(self.w) @ (Z_all - z_hat).T) + QQ

        # State and measurement cross 
        P_xz = (self.F - X_predict) @ np.diag(self.w) @ (Z_all - z_hat).T

        # Kalman gain
        K = P_xz @ np.linalg.inv(S)

        # Correct mean
        X = X_predict + K @ innovation
        X[2] = wrap2Pi(X[2])
        X = X.reshape(3)

        # Correct Convariance
        P = P_predict - K @ S @ K.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X.reshape(3))
        self.state_.setCovariance(P)


    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)      # w_0 the first weight
        self.w[1:] = 1 / (2 * (self.n + kappa))   # w_i ith weight
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state