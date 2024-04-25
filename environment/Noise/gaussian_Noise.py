import numpy as np

from .Noise import Noise


class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
        print("sigma ", sigma)

    def draw(self):
        return np.random.normal(loc=0, scale=self.sigma)
