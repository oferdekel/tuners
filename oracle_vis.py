#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
import matplotlib.pyplot as plt
from oracle import Oracle

def show1d(oracle):

    NUM_PLOTS = 4
    STD = 0.4

    fig, ax = plt.subplots(1, NUM_PLOTS, sharey=True)

    for i in range(0, NUM_PLOTS):
        
        r = np.random.randn(oracle.dim)
        r /= np.linalg.norm(r) * STD
        grid = np.linspace(-1,1,1000)

        configs = [r*x for x in grid]
        results = oracle.evaluate(configs)

        ax[i].plot(grid, results, '-', linewidth=2)

    plt.show()

def main():
    O = Oracle(2)
    show1d(O)

if __name__ == '__main__':
    main()