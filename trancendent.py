from scipy.optimize import fsolve
import numpy as np
from matplotlib import pyplot as plt

k = np.linspace(-np.pi, np.pi, 100)


def trancedent(k, a, energy_lvl):
    def equation(x, k):
        return np.cos(k*a) - np.cos(x*a) - np.sin(x*a)

    solutions = []
    for i in k:
        x0 = energy_lvl * np.pi / a
        solution = fsolve(equation, x0, args=(i,))[0]
        solutions.append(solution)

    k_shtrih = np.array(solutions)
    E = k_shtrih ** 2

    return E
