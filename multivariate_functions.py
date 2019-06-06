#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import scipy.special as sp

class MultivariateSin():

    def __init__(self, dim, num_waves = 30, pre_amplitude_scale = 1, post_amplitude_scale = 1, frequency_scale = 1):
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

    def evaluate(self, x):
        if isinstance(x, np.ndarray):
            dots = x.dot(self.vectors)
            results = [np.cos(self.phase[i] + self.frequency[i] * dots[i]) * self.pre_amplitude_scale for i in range(self.num_waves)]
            results = np.round(results)

            result = self.amplitude.dot(results)
            return result

        else:
            return [self.evaluate(element) for element in x]
