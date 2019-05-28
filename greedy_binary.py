#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateSin
from caching_oracle import CachingOracle

class GreedyBinarySearch():

    def flip_each(self):
        result = []
        for i in range(self.oracle.get_dimension()):
            copy = np.array(self.best_config)
            copy[i] = 1 - copy[i]
            result.append(copy)
        return result

    def issue_queries(self, queries):
        self.oracle.query(queries)
        best = self.oracle.get_best(0)
        self.best_config, self.best_value = best.popitem()

    def __init__(self, oracle):
        self.oracle = oracle
        self.best_config = np.zeros(oracle.get_dimension())
        self.best_value = float("inf")

        queries = self.flip_each()
        queries.append(self.best_config)
        self.issue_queries(queries)

    def step(self):
        previous_best = self.best_value
        queries = self.flip_each()
        self.issue_queries(queries)
        return previous_best - self.best_value

    def get_best(self, tolerance=0):
        return self.oracle.get_best(tolerance)


def main():
    DIM = 250
    STEPS = 10

    f = MultivariateSin(DIM)
    o = CachingOracle(f)
    s = GreedyBinarySearch(o)
    
    for i in range(STEPS):
        print(s.step())

if __name__ == '__main__':
    main()
