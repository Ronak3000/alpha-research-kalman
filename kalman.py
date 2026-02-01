import numpy as np

class KalmanRegression:
    def __init__(self):
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1000
        self.Q = np.eye(2) * 0.0001
        self.R = 0.01

    def update(self, price_y, price_x):
        H = np.array([1, price_x])
        self.P = self.P + self.Q

        y_res = price_y - np.dot(H, self.x)
        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(self.P, H.T) / S

        self.x = self.x + K * y_res
        self.P = np.dot((np.eye(2) - np.outer(K, H)), self.P)

        return self.x