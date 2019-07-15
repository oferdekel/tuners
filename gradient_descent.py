#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateLinear, MultivariateSin
from multivariate_function_cache import MultivariateFunctionCache

class GradientDescent():

    def __init__(self, oracle, kernel_radius = 0.1, step_size = 0.1, initial_point = np.empty(0)):
        self.oracle = oracle
        self.dimension = oracle.dim
        self.kernel_radius = kernel_radius 
        self.step_size = step_size

        if initial_point.size == 0:
            self.current_point = np.zeros(self.dimension)
        else:
            if len(initial_point) is not self.dimension:
                raise Exception('initial_point is not the right size')
            self.current_point = initial_point

        self.current_value = float('inf')

    def step(self, num_directions = 3):

        # sample from the Gaussian kernel
        directions = np.random.randn(num_directions, self.dimension)
        directions = np.vstack((directions, np.zeros(self.dimension)))

        # prepare queries
        queries = self.current_point + (directions * self.kernel_radius)

#        queries = np.clip(self.current_point + directions, -1, 1).tolist()
#        queries.append(self.current_point)
        results = self.oracle.evaluate(queries)
        current_value = results[-1]
        results -= current_value # subtract the value at the current point from all other values

        # estimate gradient
        directions *= results[:, None] / self.kernel_radius
        grad = directions.sum(axis=0) / num_directions

        # take gradient step
        step = grad / np.linalg.norm(grad) * self.step_size
        self.current_point = np.clip(self.current_point - step, -1, 1)

        return current_value # variable name is misleading - the function actually returns the previous value

    def get_best(self, tolerance=0):
        return self.oracle.get_best(tolerance)


def main():
    DIM = 2
    NUM_WAVES = 7
    PRE_AMPLITUDE_SCALE = 1
    POST_AMPLITUDE_SCALE = 0.45
    FREQUENCY_SCALE = 1
    ROUND = False

    STEPS = 10
    KERNEL_RADIUS = 0.1
    STEP_SIZE = 0.1

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE, ROUND)
#    f = MultivariateLinear(DIM)
    o = MultivariateFunctionCache(f)
    s = GradientDescent(o, KERNEL_RADIUS, STEP_SIZE)

    for i in range(STEPS):
        print(s.step())

if __name__ == '__main__':
    main()
