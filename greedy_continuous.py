#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateSin
from caching_oracle import CachingOracle
from random_basis import randomBasis

class GreedyContinuousSearch():

    def __init__(self, oracle, initial_point = np.empty(0)):
        self.oracle = oracle
        self.dimension = oracle.get_dimension() 
        
        if initial_point.size == 0:
            self.current_point = np.zeros(self.dimension)
        else:
            if len(initial_point) is not self.dimension:
                raise Exception('initial_point is not the right size')
            self.current_point = initial_point

        self.oracle.query([self.current_point])
        x, self.best_value = self.oracle.get_best(0).popitem()

    def step(self, directions, scale = 1, symmetrize = False):
        """ Takes a step in each of a given set of directions and commits to the step with the greatest improvement. """

        if symmetrize:
            directions = np.concatenate((directions,-directions))

        directions = directions * scale

        previous_best_value = self.best_value

        x = np.clip(self.current_point + directions, -1, 1)
        queries = x.tolist()

        self.oracle.query(queries)
        point, value = self.oracle.get_best(0).popitem()

        if value < self.best_value:
            self.best_value = value
            self.current_point = point

        return previous_best_value - self.best_value

    def step_eye(self, scale = 1):
        """ Takes a step in each of the cannonical directions and commits to the step with the greatest improvement. """

        directions = np.eye(self.dimension)
        return self.step(directions, scale, symmetrize = True)

    def step_random_orthonormal(self, scale = 1):
        """ Takes a step in each of an orthonormal set of directions and commits to the step with the greatest improvement. """

        directions = randomBasis(self.dimension)
        return self.step(directions, scale, symmetrize = True)

    def step_random(self, scale = 1, num_directions = 0):
        """ Takes a step in a given number of random directions and commits to the step with the greatest improvement. """

        if num_directions == 0:
            num_directions = self.dimension

        directions = np.random.randn(num_directions, self.dimension)
        return self.step(directions, scale, symmetrize = False)

    def get_best(self, tolerance=0):
        """ Returns the best point observed so far. """
        return self.oracle.get_best(tolerance)


def main():
    DIM = 20
    NUM_WAVES = 7
    PRE_AMPLITUDE_SCALE = 1
    POST_AMPLITUDE_SCALE = 0.45
    FREQUENCY_SCALE = 1
    STEPS = 10
    STEP_SCALE = 0.2

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE)
    o = CachingOracle(f)
    s = GreedyContinuousSearch(o, np.ones(20))

    for i in range(STEPS):
        print(s.step_random(scale = STEP_SCALE))

if __name__ == '__main__':
    main()
