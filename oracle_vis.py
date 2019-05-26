#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
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

def show1d(func, origin):
    """ Plots the optimization landscape along a few random directions. """

    GRID_SIZE = 100
    grid = np.linspace(-1, 1, GRID_SIZE)

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


def show2d(func, origin):

    GRID_SIZE = 40
    grid = np.linspace(-1, 1, GRID_SIZE)
    X, Y = np.meshgrid(grid, grid)

    fig = plt.figure()
    num_plots = func.dim // 2
    (num_cols, num_rows) = factor2(num_plots)
    perm = np.random.permutation(range(func.dim))

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
        results = results.reshape(GRID_SIZE, GRID_SIZE)

        ax = fig.add_subplot(num_rows, num_cols, k+1, projection='3d')
        ax.set_xlabel('coord ' + str(index0))
        ax.set_ylabel('coord ' + str(index1))

        ax.plot_surface(X, Y, results, cmap='hot')

    plt.tight_layout() 
    plt.show()

def main():
    DIM = 25

    f = MultivariateSin(DIM)
    origin = np.zeros(DIM)
    show2d(f, origin)

if __name__ == '__main__':
    main()