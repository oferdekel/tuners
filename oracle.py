#   Project: Tuners
#   Author: Ofer Dekel

"""Simulates an oracle that takes a list of configuration vectors and measures the speed of each one."""

from hashlib import sha1
from numpy import array, ndarray
import random

class HashableArray(ndarray):

    def __new__(cls, values):
        return array(values).view(cls)

    def __init__(self, values):
        self.__hash = int(sha1(self).hexdigest(), 16)

    def __eq__(self, other):
        return all(ndarray.__eq__(self, other))

    def __hash__(self):
        return self.__hash

    def __setitem__(self, key, value):
        raise Exception("hashable arrays are read-only")


class Oracle:

    results_cache = {}

    def __init__(self, dim):
        """ Creates an oracle of a given dimension. """
        zero_config = HashableArray([0] * dim)
        self.query({zero_config})

    def _evaluate(self, configs):
        """ Evaluates a set of configurations.
            Args:
                configs: a set of tuples
        """

        return {config: random.randint(0,100) for config in configs}

    def query(self, configs):
        """ Looks up the results for a list of configs, queries any missing configs.
            Args:
                configs: a set of tuples
        """

        to_evaluate = set(configs).difference(self.results_cache)
        results = self._evaluate(to_evaluate)
        self.results_cache.update(results)

        return {config: self.results_cache[config] for config in configs}

    def get_best(self, tolerance = 0):
        """Returns a dictionary of the configs with the smallest results observed so far, up to a specified tolerance."""

        threshold = min(self.results_cache.values()) + tolerance
        return {config:result for (config, result) in self.results_cache.items() if result <= threshold}