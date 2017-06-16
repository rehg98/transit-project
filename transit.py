#! /usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

# returns data fom specified file
def read_data(filename):
    # read in data
    t, f = np.loadtxt('data/' + str(filename) + '.txt', unpack=True)
    return  t, f

def plot_data(filename):
    t, f = read_data(filename)
    fig = plt.scatter(t,  f)
    plt.xlabel('Time from mid-transit [days]')
    plt.ylabel("Relative Flux [PPM]")
    plt.show()

def trapezoid(pars, t):
	t0, T, tau, depth = pars
	temp = t.copy()
	mask = t < t0 - T/2
	temp[mask] = 0
	mask = (t < t0 - T/2 + tau) & (t >= t0 - T/2)
	temp[mask] = depth*((t0 - T/2 + tau - t[mask])/tau - 1)
	mask = (t < t0 + T/2 - tau) & (t >= t0 - T/2 + tau)
	temp[mask] = -1 * depth
	mask = (t < t0 + T/2) & (t >= t0 + T/2 - tau)
	temp[mask] = depth * (t[mask] - t0 - T/2)/tau
	mask = t >= t0 + T/2
	temp[mask] = 0
	return temp

if __name__ == "__main__":
    #plot_data(sys.argv[1])
    pars = [0, 1.0, 0.2, 200] # t0 [center], T [duration], tau [ingress duration], depth
    t,f = read_data(7016.01)
    plt.plot(t, trapezoid(pars,t))
    plt.ylim(-250,50)
    plt.show()



