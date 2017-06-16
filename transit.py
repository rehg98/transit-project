#! /usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

# returns data fom specified file
def read_data(filename):
    # read in data
    t, f = np.loadtxt('data/' + filename, unpack=True)
    return  t, f

def plot_data(filename):
    t, f = read_data(filename)
    fig = plt.scatter(t,  f)
    plt.xlabel('Time from mid-transit [days]')
    plt.ylabel("Relative Flux [PPM]")
    plt.show()

if __name__ == "__main__":
    plot_data(sys.argv[1])



