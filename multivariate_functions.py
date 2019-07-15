#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import scipy.special as sp

class MultivariateSin():

    def __init__(self, dim, num_waves = 30, pre_amplitude_scale = 1, post_amplitude_scale = 1, frequency_scale = 1, round = True):
        self.dim = dim
        self.num_waves = num_waves
        self.pre_amplitude_scale = pre_amplitude_scale
        self.round = round

        # draw random probability vectors
        self.vectors = np.random.randn(dim, num_waves)
        self.vectors /= np.linalg.norm(self.vectors, axis = 0) 
        
        self.amplitude = post_amplitude_scale * np.random.randn(num_waves)
        self.frequency = frequency_scale * np.random.randn(num_waves) * np.sqrt(dim)
        self.phase = 36000 * np.random.randn(num_waves) * np.sqrt(dim)

    def evaluate(self, X):
        dots = X.dot(self.vectors) * self.frequency + self.phase
        results = np.cos(dots) + self.pre_amplitude_scale
        
        if self.round is True:
            results = np.round(results)

        result = results.dot(self.amplitude)
        return result


class MultivariateLinear():

    def __init__(self, dim):
        self.w = np.random.randn(dim).T
        self.dim = dim

    def evaluate(self, X):
        return X.dot(self.w)
