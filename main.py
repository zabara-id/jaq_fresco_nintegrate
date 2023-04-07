from scipy.integrate import quad
import numpy as np
from scipy.special import erf
from scipy import inf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# инициализация параметров
x0 = 1
sigma = 0.5
kappa = 1
h_pr = 1
m = 1
E_linked = - np.power(kappa * h_pr, 2) / (2 * m)
B = 0

def psi_0(x: float) -> float:
    return np.exp(-np.power((x - x0) / (2 * sigma), 2)) / (np.power(2 * np.pi * sigma * sigma, 1 / 4))


def psi_linked(x: float) -> float:
    return np.sqrt(kappa) * np.exp(-kappa * np.abs(x))


def psi_chet(E: float, x: float) -> float:
    k = np.sqrt(2 * m * E) / h_pr
    return B * (np.cos(k * x) - (kappa / k) * np.sin(k * np.abs(x)))


def psi_nechet(E: float, x: float) -> float:
    k = np.sqrt(2 * m * E) / h_pr
    return np.sin(k * x) / np.sqrt(np.pi)


def c_linked(by_integral=False) -> float:
    if by_integral:
        def inter(x):
            return psi_linked(x) * psi_0(x)

        return quad(inter, -inf, inf)[0]
    else:
        res = -(np.power(np.pi / 2, 1 / 4) * np.sqrt(kappa * sigma)) * \
              np.exp(np.power(kappa * sigma, 2) - kappa * x0) * \
              (np.exp(2 * kappa * x0) * erf((x0 + 2 * kappa * np.power(sigma, 2)) / (2 * sigma)) - erf(
                  (x0 + 2 * kappa * np.power(sigma, 2)) / (2 * sigma)) - np.exp(2 * kappa * x0) - 1)
        return res


def c_nechet(E: float, by_integral=False) -> float:
    if by_integral:
        def inter(x):
            return psi_nechet(E, x) * psi_0(x)

        return quad(inter, -inf, inf)[0]
    else:
        k = np.sqrt(2 * m * E) / h_pr
        res = (np.power(2 / np.pi, 1 / 4) * np.sqrt(2 * sigma)) * \
              np.sin(k * x0) * np.exp(-np.power(k * sigma, 2))
        return res


def c_chet(E: float, by_integral=False) -> float:
    if by_integral:
        def inter(x):
            return psi_chet(E, x) * psi_0(x)

        return quad(inter, -inf, inf)[0]
    else:
        k = np.sqrt(2 * m * E) / h_pr
        arg1 = complex(real=x0, imag=2 * k * np.power(sigma, 2)) / (2 * sigma)
        arg2 = complex(real=x0, imag=-2 * k * np.power(sigma, 2)) / (2 * sigma)
        res = - B * (np.power(np.pi / 2, 1 / 4) * np.sqrt(sigma) / k) * \
              np.exp(-np.power(k * sigma, 2)) * (kappa * complex(real=np.sin(k * x0), imag=-np.cos(k * x0)) *
                                                 erf(arg1) + kappa * complex(real=np.sin(k * x0), imag=np.cos(k * x0)) *
                                                 erf(arg2) - 2 * k * np.cos(k * x0))
        return res


# функция для интегрирования комплексных функций
def complex_quad(func, a, b):
    def real_func(x):
        return np.real(func(x))

    def imag_func(x):
        return np.imag(func(x))

    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)

    return complex(real=real_integral[0], imag=imag_integral[0])


def psi(x: float, t: float) -> complex:
    s1 = c_linked() * psi_linked(x) * np.exp(complex(real=0, imag=-E_linked * t / h_pr))

    def nechet_intergal(E):
        return c_nechet(E) * psi_nechet(E, x) * np.exp(complex(real=0, imag=-E * t / h_pr))

    def chet_intergal(E):
        return c_chet(E) * psi_chet(E, x) * np.exp(complex(real=0, imag=-E * t / h_pr))

    s2 = complex_quad(nechet_intergal, 0, inf)
#    s3 = complex_quad(chet_intergal, 0.00001, inf)
    return s1 + s2 #+ s3


x = np.linspace(0, 10, dtype=float)
t = np.linspace(-10, 10, dtype=float)


def prob(t):
    W = np.zeros_like(x, dtype=complex)
    for i in range(W.shape[0]):
        W[i] = psi(x[i], 0)

    return (np.abs(np.fft.fft(W))) ** 2

#
# W = prob(0)
# N = len(W)
# dt = 0.1
# xf = np.fft.fftfreq(N, dt)
#
# fig, ax = plt.subplots()
#
# ax.plot(xf, W, linewidth=2.0)
#
# plt.show()





