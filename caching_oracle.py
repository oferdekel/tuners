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


class CachingOracle:
    """ Simulates an oracle that evaluates sets of configuration vectors. """

    def __init__(self, function):
        self.function = function
        self.results_cache = {}

    def __str__(self):
        return '\n'.join(str(config) + '\t' + str(result) for (config, result) in self.results_cache.items())

    def query(self, configs):
        """ Looks up the results for a list of configs, queries any missing configs.
            Args:
                configs: a set of tuples
        """
        configs = {HashableArray(config) for config in configs}
        new_configs = list(set(configs).difference(self.results_cache))    # get a (unique) set of configs that aren't in the cache
        new_results = self.function.evaluate(new_configs)    # evaluate the new configs
        self.results_cache.update(zip(new_configs, new_results))    # cache the new results
        return {config: self.results_cache[config] for config in configs}    # return a dictionary of results, both new and old

    def get_best(self, tolerance = 0):
        """Returns a dictionary of the configs with the smallest results observed so far, up to a specified tolerance."""
        threshold = min(self.results_cache.values()) + tolerance
        return {config:result for (config, result) in self.results_cache.items() if result <= threshold}

    def get_dimension(self):
        return self.function.dim



def main():
    f = MultivariateSin(10)
    o = CachingOracle(f)
    o.query([np.ones(10), np.zeros(10)])

if __name__ == '__main__':
    main()