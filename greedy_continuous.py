#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateSin
from caching_oracle import CachingOracle
from random_basis import randomBasis

class GreedyContinuousSearch():

    def __init__(self, oracle):
        self.oracle = oracle
        self.dimension = oracle.get_dimension() 
        self.best_config = np.zeros(oracle.get_dimension())
        self.oracle.query([self.best_config])
        x, self.best_value = self.oracle.get_best(0).popitem()

    def step(self, directions = np.empty(0)):
        """ Takes a step in each of a given set of directions and commits to the step with the greatest improvement. """

        if directions.size == 0:
            I = np.eye(self.dimension)
            directions = np.concatenate((I,-I))

        previous_best_value = self.best_value

        x = np.clip(self.best_config + directions, -1, 1)
        queries = x.tolist()

        self.oracle.query(queries)
        config, value = self.oracle.get_best(0).popitem()

        if value < self.best_value:
            self.best_value = value
            self.best_config = config

        return previous_best_value - self.best_value

    def get_best(self, tolerance=0):
        return self.oracle.get_best(tolerance)


def main():
    DIM = 20
    NUM_WAVES = 7
    PRE_AMPLITUDE_SCALE = 1
    POST_AMPLITUDE_SCALE = 0.45
    FREQUENCY_SCALE = 1
    STEPS = 10

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE)
    o = CachingOracle(f)
    s = GreedyContinuousSearch(o)


    for i in range(STEPS):
        b = randomBasis(DIM)
        directions = np.concatenate((b,-b))
        print(s.step(directions))

if __name__ == '__main__':
    main()
