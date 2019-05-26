#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import scipy.special as sp

class MultivariateSin():

    def __init__(self, dim, num_waves = 30, frequency_scale = 1):
        self.dim = dim
        self.num_waves = num_waves
        
        # draw random probability vectors
        self.vectors = np.random.randn(dim, num_waves)
        self.vectors /= np.linalg.norm(self.vectors, axis = 0) 
        
        self.magnitude = np.random.randn(num_waves)
        self.frequency = frequency_scale * np.random.randn(num_waves) * np.sqrt(dim)
        self.phase = 360 * np.random.randn(num_waves) * np.sqrt(dim)

    def evaluate(self, x):

        if isinstance(x, np.ndarray):
            dots = x.dot(self.vectors)
            results = [np.cos(self.phase[i] + self.frequency[i] * dots[i]) for i in range(self.num_waves)]
            result = self.magnitude.dot(results)
            return result

        else:
            return [self.evaluate(element) for element in x]
