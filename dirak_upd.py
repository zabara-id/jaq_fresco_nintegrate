import numpy as np
from numpy import exp, pi, sqrt
import matplotlib.pyplot as plt
from scipy.special import erf
from numpy.linalg import eig
from trancendent import trancedent
import warnings
from matplotlib import MatplotlibDeprecationWarning

N_int = 1
sgm = 0.01
a = 4
h = 1
m = 1


def vn(n):
    return -N_int * sqrt(pi) * sigma / 2 / a * exp(-pi ** 2 * n ** 2 * sigma ** 2 / a ** 2) * \
           (erf((2 * 1j * pi * n * sigma ** 2 + a ** 2) / (2 * a * sigma)) -
            erf((2 * 1j * pi * n * sigma ** 2 - a ** 2) / (2 * a * sigma)))


def get_A(n, k):
    diag = h ** 2 / 2 / m * (2 * pi * n / a) ** 2 + h ** 2 * k / m * (2 * pi * n / a) + h ** 2 * k ** 2 / 2 / m
    return np.diag(diag)


p = 0.999 / (1e-8 - 1)
pp = 0.01 - p
sigma = p * sgm + pp
if sigma > 0.1:
    sigma = 0.1


def main():
    global N_int, a, h, m, sigma

    N_int *= 10 ** -1

    N = 40

    n = np.arange(-N, N + 1)

    V = np.zeros((2 * N + 1, 2 * N + 1), dtype=np.complex128)
    V_ = np.zeros((2 * N + 1, 2 * N + 1))

    for i, vali in enumerate(range(-N, N + 1)):
        for m, valm in enumerate(range(-N, N + 1)):
            V[i, m] = vn(vali - valm)
            V_[i, m] = (vali - valm)

    res = []
    ks = []

    for k in np.linspace(-pi / a, pi / a, 101):
        A = get_A(n, k)
        M = A + V
        eigs = np.real(eig(M)[0])
        res.append(np.sort(eigs))
        ks.append(k)

    res = np.array(res)
    k = np.array(ks)

    energy_level = 1

    analytical_solution = trancedent(k, a, energy_level)
    numerical_solution = np.abs(res[:, energy_level + 1])

    min_num = np.min(numerical_solution)
    max_num = np.max(numerical_solution)

    min_an = np.min(analytical_solution)
    max_an = np.max(analytical_solution)

    p = (min_num - max_num) / (min_an - max_an)
    pp = min_num - min_an * p

    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    plt.style.use('dark_background')

    plt.plot(k, numerical_solution, label='numerical {}'.format(energy_level))
    plt.plot(k, analytical_solution * p + pp, label='analytical {}'.format(energy_level))

    # plt.plot(k, np.abs(res[:, energy_level + 2]), label='numerical {}'.format(energy_level+1))
    # plt.plot(k, trancedent(k, a, energy_level+1), label='analytical {}'.format(energy_level+1))
    #
    # plt.plot(k, np.abs(res[:, energy_level + 3]), label='numerical {}'.format(energy_level+2))
    # plt.plot(k, trancedent(k, a, energy_level+2), label='analytical {}'.format(energy_level+2))

    plt.title(f"$E_{energy_level}$")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
