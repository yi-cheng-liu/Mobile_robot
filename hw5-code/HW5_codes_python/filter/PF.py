
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

from scipy.stats import multivariate_normal
from numpy.random import default_rng
rng = default_rng()

class PF:
    # PF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):
        np.random.seed(2)
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        # PF parameters
        self.n = init.n
        self.Sigma = init.Sigma
        self.particles = init.particles
        self.particle_weight = init.particle_weight

        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        ###############################################################################
        # TODO: Implement the prediction step for PF, remove pass                     #
        # Hint: Propagate your particles. Particles are saved in self.particles       #
        # Hint: Use rng.standard_normal instead of np.random.randn.                   #
        #       It is statistically more random.                                      #
        ###############################################################################
        # Propagate particles through the motion model
        for i in range(self.n):
            self.particles[:, i] = self.gfun(self.particles[:, i], u + np.sqrt(self.M(u)) @ rng.standard_normal(3))
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


    def correction(self, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        
        ###############################################################################
        # TODO: Implement the correction step for PF                                  #
        # Hint: self.mean_variance() will update the mean and covariance              #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        # Two landmarks
        landmark1_x, landmark1_y = landmark1.getPosition()[0], landmark1.getPosition()[1]
        landmark2_x, landmark2_y = landmark2.getPosition()[0], landmark2.getPosition()[1]

        # Update particle weights based on measurements
        for i in range(self.n):
            # Predicted measurement
            z_hat1 = self.hfun(landmark1_x, landmark1_y, self.particles[:, i])
            z_hat2 = self.hfun(landmark2_x, landmark2_y, self.particles[:, i])

            prob1 = multivariate_normal.pdf(z[ :2], mean=z_hat1, cov=self.Q)
            prob2 = multivariate_normal.pdf(z[3:5], mean=z_hat2, cov=self.Q)

            self.particle_weight[i] *= prob1 * prob2
        
        # Normalize the importance weight
        self.particle_weight /= np.sum(self.particle_weight)

        # Measure the degeneracy using the effective sample size (1 < neff < n)
        neff = 1 / np.sum(self.particle_weight**2)
        
        # Use resample algorithm with higher weights (n/3 is the resample threshold)
        if neff < self.n / 3:
            self.resample()
        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.mean_variance()


    def resample(self):
        new_samples = np.zeros_like(self.particles)
        new_weight = np.zeros_like(self.particle_weight)
        W = np.cumsum(self.particle_weight)
        r = np.random.rand(1) / self.n
        count = 0
        for j in range(self.n):
            u = r + j/self.n
            while u > W[count]:
                count += 1
            new_samples[:,j] = self.particles[:,count]
            new_weight[j] = 1 / self.n
        self.particles = new_samples
        self.particle_weight = new_weight
    

    def mean_variance(self):
        X = np.mean(self.particles, axis=1)
        sinSum = 0
        cosSum = 0
        for s in range(self.n):
            cosSum += np.cos(self.particles[2,s])
            sinSum += np.sin(self.particles[2,s])
        X[2] = np.arctan2(sinSum, cosSum)
        zero_mean = np.zeros_like(self.particles)
        for s in range(self.n):
            zero_mean[:,s] = self.particles[:,s] - X
            zero_mean[2,s] = wrap2Pi(zero_mean[2,s])
        P = zero_mean @ zero_mean.T / self.n
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)
    
    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

