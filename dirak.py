import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Задание параметров функции и пределов интегрирования
sigma = 2
a = 0.1
inf = np.inf
pi = np.pi
hbar = 1
m = 1

# Определение функции
def integrand(x, a, n, sigma):
    return np.exp(-x ** 2 / sigma ** 2) * np.exp(-2 * np.pi * 1j * n * x / a)


# Заполняем матрицу V
def v_matrix(n: int) -> np.ndarray:
    V = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n - i):
            V[j, j + i] = quad(integrand, -inf, inf, args=(a, i, sigma))[0]
            V[j + i, j] = quad(integrand, -inf, inf, args=(a, -i, sigma))[0]

    return V


# # Заполняем матрицу A
# def a_matrix(n: int):
#     for i in range(0, n):
#         for k in np.linspace(-pi/a, pi/a, 100):
#             A[i, i] = hbar**2 / 2*m * (2 * pi * i / a)**2 + hbar**2 * k**2 / m * (2*pi*i/a)
#
#     return A


def energy(n: int, num=100) -> np.ndarray:
    E = np.zeros((n, num))
    temp = 0
    for k in np.linspace(-pi/a, pi/a, num):
        print(temp, "%", sep='')
        V = np.zeros((n, n))
        A = np.zeros((n, n))
        e = np.zeros(num)
        # Вычисляем матрицу V
        for i in range(0, n):
            for j in range(0, n - i):
                V[j, j + i] = quad(integrand, -1000, 1000, args=(a, i, sigma))[0]
                V[j + i, j] = quad(integrand, -1000, 1000, args=(a, -i, sigma))[0]
        # Вычисляем матрицу A для конкретного k
        for i in range(0, n):
            A[i, i] = hbar ** 2 / (2*m) * (2 * pi * i / a) ** 2 + hbar ** 2 * k ** 2 / m * (2 * pi * i / a) + hbar**2 * k**2 / (2*m)

        # Диагонализируем матрицу (A+V)
        eigenvalues = np.sort(np.linalg.eig(V + A)[0])

        for i in range(0, n):
            E[i,temp] = eigenvalues[i]

        temp += 1


    return E



# определение матрицы E
# E = energy(5)
# создание оси x
# x = np.linspace(-np.pi/a, np.pi/a, 100)
#
# # построение линий для каждой строчки матрицы E
# for i in range(E.shape[0]):
#     y = E[i,:]
#     plt.plot(x, y)
#
# # добавление легенды и заголовка графика
# plt.legend()
# plt.title('E_n(k)')
# plt.grid(True)
#
# # показ графика
# plt.show()

print(v_matrix(10))

# V = v_matrix(4)
# A = a_matrix(4)
# print(V)
# print(A)


# Вычисление интеграла
# result, abserror = quad(integrand, -inf, inf, args=(a, n, sigma))

# print("Результат интегрирования:", result)
# print("Абсолютная ошибка:", abserror)
