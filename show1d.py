#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multivariate_functions import MultivariateSin
from utils import factor2

def show1d(func, origin, grid_size = 100):
    """ Plots the optimization landscape along a few random directions. """
    
    grid = np.linspace(-1, 1, grid_size).T

    (num_cols, num_rows) = factor2(func.dim)
    fig, ax = plt.subplots(num_rows, num_cols, sharey=True)

    for k in range(func.dim):
        
        queries = np.tile(origin, (grid_size, 1))
        queries[:, k] = grid
        results = func.evaluate(queries)
        
        i, j = divmod(k, num_cols)
        
        if isinstance(ax[i], np.ndarray):
            current = ax[i][j]
        else:
            current = ax[j]    # if dim is prime, there is only one row of axes

        current.plot(grid, results, '-', linewidth=2)
        current.set_xlabel('coord ' + str(k))

    plt.tight_layout() 
    plt.show()


def main():
    DIM = 20
    NUM_WAVES = 7
    PRE_AMPLITUDE_SCALE = 1
    POST_AMPLITUDE_SCALE = 0.45
    FREQUENCY_SCALE = 1

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE)
    origin = np.zeros(DIM)

    GRID_SIZE = 50
    show1d(f, origin, GRID_SIZE)

if __name__ == '__main__':
    main()