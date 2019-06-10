#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateSin
from caching_oracle import CachingOracle

class GradientDescent():

    def __init__(self, oracle, kernel_radius = 0.1, step_size = 0.1):
        self.oracle = oracle
        self.num_directions = 2 * oracle.get_dimension()
        self.kernel_radius = kernel_radius 
        self.step_size = step_size
        self.current = np.zeros(oracle.get_dimension())

    def step(self, basis = None):

        # prepare queries
        scaled_basis = basis * self.kernel_radius
        query_directions = np.concatenate((scaled_basis, -scaled_basis), axis=0)
        queries = np.clip(query_directions, -1, 1).tolist()   # ensure all queries are in the cube
        queries.append(self.current)

        results = self.oracle.query(queries)
        results.pop()
        results = np.array(results)
        
        # estimate gradient
        query_directions *= results[:, None]
        grad = query_directions.sum(axis=0)
        step = grad / np.linalg.norm(grad) * self.step_size

        # perform step and project back onto the cube
        self.current = np.clip(self.current - step, -1, 1)
        return self.current

    def get_best(self, tolerance=0):
        return self.oracle.get_best(tolerance)


def main():
    DIM = 20
    NUM_WAVES = 7
    PRE_AMPLITUDE_SCALE = 1
    POST_AMPLITUDE_SCALE = 0.45
    FREQUENCY_SCALE = 1
    STEPS = 100
    KERNEL_RADIUS = 0.5
    STEP_SIZE = 0.2

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE)
    o = CachingOracle(f)
    s = GradientDescent(o, KERNEL_RADIUS, STEP_SIZE)
    basis = np.eye(DIM)

    for i in range(STEPS):
        print(s.step(basis))
        print(s.get_best())

if __name__ == '__main__':
    main()
