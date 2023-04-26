import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.integrate import quad


x0 = 1
sigma = 0.5
kappa = 1
h_pr = 0.5
m = 0.01
E_linked = - np.power(kappa * h_pr, 2) / (2 * m)


def c_linked() -> float:
    res = -(np.power(np.pi / 2, 1 / 4) * np.sqrt(kappa * sigma)) * \
          np.exp(np.power(kappa * sigma, 2) - kappa * x0) * \
          (np.exp(2 * kappa * x0) * erf((x0 + 2 * kappa * np.power(sigma, 2)) / (2 * sigma)) - erf(
              (x0 - 2 * kappa * np.power(sigma, 2)) / (2 * sigma)) - np.exp(2 * kappa * x0) - 1)
    return res


print(c_linked())


p = np.linspace(-3, 3, 100)


def I1(p):
    I1_abs = np.power(np.abs(c_linked() * 2 * np.power(kappa, 1.5) / (np.power(kappa, 2) + np.power(p, 2) / np.power(h_pr, 2))), 2)
    return I1_abs


def I1_normed(p):
    I1_abs_normed = np.power(np.abs(c_linked() * 2 * np.power(kappa, 1.5) / (np.power(kappa, 2) + np.power(p, 2) / np.power(h_pr, 2))), 2) / quad(I1, -2, 2)[0]
    return I1_abs_normed


a = quad(I1, -2, 2)[0]
I1_ans = I1(p) / a
print(quad(I1_normed, -2, 2)[0])
# Построение графика
plt.plot(p, I1_ans)
plt.title("kappa = 1")
plt.xlabel("Ось p")
plt.ylabel("Ось W_p")
plt.grid(True)

# Отображение графика
plt.show()