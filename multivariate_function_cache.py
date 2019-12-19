#   Project: Tuners
#   Author: Ofer Dekel

from hashlib import sha1
import numpy as np
from multivariate_functions import MultivariateSin

class HashableArray(np.ndarray):
    """ Extends numby ndarrays by making them immutable and hashable, so that they can be dictionary keys. """

    def __new__(cls, values):
        return np.array(values).view(cls)

    def __init__(self, values):
        self.__hash = int(sha1(self).hexdigest(), 16)

    def __eq__(self, other):
        return all(np.ndarray.__eq__(self, other))

    def __hash__(self):
        return self.__hash

    def __setitem__(self, key, value):
        raise Exception("hashable arrays are read-only")    # make the array immutable


class MultivariateFunctionCache:
    """ Simulates an oracle that evaluates sets of configuration vectors. """

    def __init__(self, function):
        self.function = function
        self.dim = function.dim
        self.results_cache = {}

    def __str__(self):
        return '\n'.join(str(config) + '\t' + str(result) for (config, result) in self.results_cache.items())

    def evaluate(self, X):
        """ Looks up the results for a given X, only queries the underlying function for missing points.
            Args:
                X: a numpy matrix, where each row is a query
        """
        
        if isinstance(X, np.ndarray):
            queries = [HashableArray(X)]
        else:
            queries = [HashableArray(x) for x in X]
        
        new_queries = set(queries).difference(self.results_cache)    # get a (unique) set of configs that aren't in the cache
        new_queries_matrix = np.stack(list(new_queries))    # BUG 0- creates tensr fos shape (1,20,20)
        new_results = self.function.evaluate(new_queries_matrix)    # evaluate the new configs
        self.results_cache.update(zip(new_queries, new_results))    # cache the new results
        return np.stack([self.results_cache[x] for x in queries])   # return a matrix of results that matches the order of the queries

    def get_best(self, tolerance = 0):
        """Returns a dictionary of the configs with the smallest results observed so far, up to a specified tolerance."""
        threshold = min(self.results_cache.values()) + tolerance
        return {config:result for (config, result) in self.results_cache.items() if result <= threshold}

def main():
    f = MultivariateSin(10)
    o = MultivariateFunctionCache(f)
    o.query([np.ones(10), np.zeros(10)])

if __name__ == '__main__':
    main()