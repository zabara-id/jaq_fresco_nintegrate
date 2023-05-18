import numpy as np
import matplotlib.pyplot as plt

# Количество электронов
n_electrons = 1000

# Генерация случайных спинов электронов
spins = np.random.choice([-1, 1], size=n_electrons)

# Магнитное поле
B = 1.0

# Координаты детекторов
detectors = np.array([-1, 1])

# Результаты обнаружения спина электрона
results = np.zeros(2)

# Прохождение электронов через магнитное поле и детекторы
for spin in spins:
    # Изменение направления спина электрона под действием магнитного поля
    spin *= np.random.choice([-1, 1], p=[0.5, 0.5])

    # Обнаружение спина электрона
    if spin > 0:
        results[0] += 1
    else:
        results[1] += 1

# Отображение результатов в виде графика
plt.bar(detectors, results / n_electrons)
plt.xticks(detectors)
plt.xlabel('Направление спина')
plt.ylabel('Вероятность обнаружения')
plt.title('Опыт Штерна-Герлаха')
plt.show()
