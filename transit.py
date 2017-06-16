#! /usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# returns data fom specified file
def read_data(filename):
    # read in data
    t, f = np.loadtxt('data/' + str(filename) + '.txt', unpack=True)
    return  t, f

def plot_data(filename):
    t, f = read_data(filename)
    fig = plt.scatter(t,  f, color='k', s=6)
    plt.xlabel('Time from mid-transit [days]')
    plt.ylabel("Relative Flux [PPM]")
    #plt.show()

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

def vary_depth(depths):
	t = np.linspace(-2., 2., 1000)
	for d in depths:
	 plt.plot(t, trapezoid([0., 1., 0.2, d], t))
	plt.show()

def vary_duration(durations):
	t = np.linspace(-2., 2., 1000)
	for d in durations:
	 plt.plot(t, trapezoid([0., d, 0.2, 200], t))
	plt.show()

def vary_tau(taus):
	t = np.linspace(-2., 2., 1000)
	for tau in taus:
	 plt.plot(t, trapezoid([0., 1., tau, 200], t))
	plt.show()

def vary_t0(t0s):
	t = np.linspace(-2., 2., 1000)
	for t0 in t0s:
	 plt.plot(t, trapezoid([t0, 1., 0.2, 200], t))
	plt.show()

def plot_fit(file, param_guess):
	t,f = read_data(file)
	plt.subplot(211)
	plt.plot(t, trapezoid(param_guess,t), c='r')
	plot_data(file)
	plt.subplot(212)
	f = f - trapezoid(param_guess,t)
	fig = plt.scatter(t,  f, color='k', s=6)
	fig = plt.plot([t[0],t[-1]],  [0,0], color='lime')
	plt.xlabel('Time from mid-transit [days]')
	plt.ylabel("Residuals")

	plt.tight_layout()
	plt.show()

def __obj_func(pars, t, f):
	return np.sum((f - trapezoid(pars, t))**2)

def fit_trapezoid(file, method = 'Nelder-Mead'):
	t,f = read_data(file)
	return scipy.optimize.minimize(__obj_func, [0, 0.8, 0.2, 200], args=(t, f), method = method)

if __name__ == "__main__":
    #plot_data(sys.argv[1])
    """
    pars = [0, 1.0, 0.2, 200] # t0 [center], T [duration], tau [ingress duration], depth
    t,f = read_data(7016.01)
    plt.plot(t, trapezoid(pars,t))
    plt.ylim(-250,50)
    plt.show()
    """
    fit = fit_trapezoid(7016.01, method = 'Nelder-Mead')
    print(fit)
    plot_fit(7016.01, fit.x)



