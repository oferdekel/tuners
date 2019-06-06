#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from multivariate_functions import MultivariateSin

def replace_coord(arr, index, value):
    copy = np.array(arr)
    copy[index] = value
    return copy

def factor2(num):
    """ Returns integers (a,b) such that a >= b, a and b are as close as possible, and a * b = num. """

    a = int(np.ceil(np.sqrt(num)))
    while num % a > 0:
        a += 1 
    b = int(num / a)
    return (a, b)

def show1d(func, origin, grid_size = 100):
    """ Plots the optimization landscape along a few random directions. """
    
    grid = np.linspace(-1, 1, grid_size)

    (num_cols, num_rows) = factor2(func.dim)
    fig, ax = plt.subplots(num_rows, num_cols, sharey=True)

    for k in range(func.dim):
        points = [replace_coord(origin, k, val) for val in grid]            
        results = func.evaluate(points)
        
        i, j = divmod(k, num_cols)
        
        if isinstance(ax[i], np.ndarray):
            current = ax[i][j]
        else:
            current = ax[j]    # if dim is prime, there is only one row of axes

        current.plot(grid, results, '-', linewidth=2)
        current.set_xlabel('coord ' + str(k))

    plt.tight_layout() 
    plt.show()


def show2d(func, origin, grid_size = 40):

    grid = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(grid, grid)

    fig = plt.figure()
    num_plots = func.dim // 2
    (num_cols, num_rows) = factor2(num_plots)
    perm = np.random.permutation(range(func.dim))

    all_results = []
    max_result = 0
    min_result = 0

    for k in range(num_plots):
        index0 = perm[2 * k]
        index1 = perm[2 * k + 1]
        
        queries = []
        for x in grid:
            for y in grid:
                query = np.array(origin) 
                query[index0] = x
                query[index1] = y
                queries.append(query)

        results = np.array(func.evaluate(queries))
        max_result = max(max(results), max_result)
        min_result = min(min(results), min_result)
        results = results.reshape(grid_size, grid_size)
        all_results.append(results)

    for k in range(num_plots):
        #clrs.Normalize(vmin = min_result, vmax=max_result)
        ax = fig.add_subplot(num_rows, num_cols, k+1, projection='3d')
        ax.set_zlim3d(min_result, max_result)
        ax.set_xlabel('coord ' + str(index0))
        ax.set_ylabel('coord ' + str(index1))
        ax.plot_surface(X, Y, all_results[k], cmap='hot', vmin=min_result, vmax=max_result)

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
    show2d(f, origin, GRID_SIZE)

if __name__ == '__main__':
    main()