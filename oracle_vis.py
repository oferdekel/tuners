#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from oracle import Oracle

def replace_coord(arr, index, value):
    copy = np.array(arr)
    copy[index] = value
    return copy

def show1d(oracle, config):
    """ Plots the optimization landscape along a few random directions. """

    GRID_SIZE = 1000
    NUM_PLOT_ROWS = 3
    NUM_PLOT_COLS = 4
 
    grid = np.linspace(-1, 1, GRID_SIZE)
    fig, ax = plt.subplots(NUM_PLOT_ROWS, NUM_PLOT_COLS, sharey=True)

    for i in range(NUM_PLOT_ROWS):
        for j in  range(NUM_PLOT_COLS):

            k = np.random.randint(0, oracle.dim)

            configs = [replace_coord(config, k, v) for v in grid]            
            results = oracle.evaluate(configs)

            ax[i][j].plot(grid, results, '-', linewidth=2)
            ax[i][j].set_xlabel('coord ' + str(k))

    plt.tight_layout() 
    plt.show()


def show2d(oracle, config):

    GRID_SIZE = 100
    NUM_PLOT_ROWS = 3
    NUM_PLOT_COLS = 4

    fig = plt.figure()
    
    grid = np.linspace(-1, 1, GRID_SIZE)
    X, Y = np.meshgrid(grid, grid)

    for i in range(1, NUM_PLOT_ROWS * NUM_PLOT_COLS + 1):
        
        perm = np.random.permutation(range(oracle.dim))
        index0 = perm[0]
        index1 = perm[1]
        
        queries = []
        for x in grid:
            for y in grid:
                query = np.array(config) 
                query[index0] = x
                query[index1] = y
                queries.append(query)

        results = np.array(oracle.evaluate(queries))
        results = results.reshape(GRID_SIZE, GRID_SIZE)

        ax = fig.add_subplot(NUM_PLOT_ROWS, NUM_PLOT_COLS, i, projection='3d')
        ax.set_xlabel('coord ' + str(index0))
        ax.set_ylabel('coord ' + str(index1))

        ax.plot_surface(X, Y, results, cmap='hot')

    plt.tight_layout() 
    plt.show()

def main():
    DIM = 10

    O = Oracle(DIM)
    config = np.zeros(DIM)
    show2d(O, config)

if __name__ == '__main__':
    main()