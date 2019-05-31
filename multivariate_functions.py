#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import scipy.special as sp

class MultivariateSin():

    def __init__(self, dim, num_waves = 30, amplitude_scale = 1, frequency_scale = 1, round = False):
        self.dim = dim
        self.num_waves = num_waves
        self.round = round

        # draw random probability vectors
        self.vectors = np.random.randn(dim, num_waves)
        self.vectors /= np.linalg.norm(self.vectors, axis = 0) 
        
        self.amplitude = amplitude_scale * np.random.randn(num_waves)
        self.frequency = frequency_scale * np.random.randn(num_waves) * np.sqrt(dim)
        self.phase = 36000 * np.random.randn(num_waves) * np.sqrt(dim)

    def evaluate(self, x):
        if isinstance(x, np.ndarray):
            dots = x.dot(self.vectors)
            results = [np.cos(self.phase[i] + self.frequency[i] * dots[i]) for i in range(self.num_waves)]
            result = self.amplitude.dot(results)
            
            if self.round:
                return round(result)
            else:
                return result

        else:
            return [self.evaluate(element) for element in x]
