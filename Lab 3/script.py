

# ЛР № 3
# Применение однослойной нейронной сети с линейной функцией
# активации для прогнозирования временных рядов

import numpy as np
import matplotlib.pyplot as plt

def X(T): # возвращает результат моделируемой функции от значения времени
    return [np.sin(t) ** 2  for t in T]

N = 20 # количество точек

a = 0 # интервал
b = 2

T = list(np.linspace(a, 2 * b - a, 2 * N))
RightX = X(T)
TryX = [0 for i in range(N)]


def DeltaW(x, q, n): # находит величину, на которую изменятся Wi, для пороговой ФА
    return n * q * x

def Net(x, w): # находит значение сетевого входа НС
    return sum([w_i * x_i for w_i, x_i in zip(w, x)])

def MeanSquareError(RightX, TryX, p):
    summa = 0
    for rx_i, tx_i in zip(RightX[p:], TryX[p:]):
        summa += (rx_i - tx_i) ** 2
    return summa ** 0.5

def Learning(p, n, m): # обучение НС методом скользящего окна

    for k in range(p): TryX[k] = RightX[k]
    w = [0] * p
    era = 0

    while(era < m):

        for l in range(p, N): # 16 шагов эпохи

            TryX[l] = Net(RightX[l - p:l-1], w)
            q = RightX[l] - TryX[l]
            for k in range(0, p):
                w[k] += DeltaW(RightX[l - p + k], q, n)

        era += 1

        # print("\nera = ",  np.round(era, 3))
        # print("TryX : ", np.around(TryX, 3))
        # print("w : ", np.around(w, 3))
        # print("e = ", np.round(e, 3))

    print(np.around(TryX, 3))
    return list(TryX), w

def Graph(TryX, p, arg = "", name = ""): # строит 2 графика : исходной и полученной функции в зависмости от временного ряда
    fig, ax = plt.subplots()
    ax.plot(T, RightX, 'bo-', linewidth=3, markersize=5)
    ax.plot(T, TryX, 'ro-', linewidth=2, markersize=3)
    plt.title("X(t)\n" + name + str(arg))
    plt.xlabel('t')
    plt.ylabel('X')
    plt.axvline(x=a,  linestyle='--')
    plt.axvline(x=b, linestyle='--')
    plt.axvline(x=T[p], linestyle='--', color = 'g')
    plt.grid(True)
    plt.show()

def Graph_E(e, arg, name):  # строит график зависимости ошибки от arg : n, p, m
    plt.plot(arg, e, 'bo-', linewidth=2, markersize=5)
    plt.title("E(" + name + ")")
    plt.xlabel(name)
    plt.ylabel('E')
    plt.grid(True)
    plt.show()

def Forecast(E, n, p, m):
    TryX, w = Learning(p, n, m)
    TryX.extend(np.zeros(N))
    for l in range(N, 2 * N):
        TryX[l] = Net(RightX[l - p: l - 1], w)
    E.append(MeanSquareError(RightX[N:], TryX[N:], p))
    return TryX


if __name__=="__main__":

    med_n = 0.3  # норма обучения
    range_n = np.around(np.linspace(0.1, 1, 10), 1)

    med_p = 4 # размер "окна" данных
    range_p = range(1, 17)

    med_m = 1000 # количество эпох
    range_m = range(400, 5001, 100)

    E = []

    for p in range_p:    # исследование относительно размера окна
        print("\n\np = ", p)
        Forecast(E, med_n, p, med_m)
    Graph_E(E, range_p, "p")

    E.clear()

    for n in range_n:  # исследование относительно нормы обучения
        print("\n\nn = ", n)
        Forecast(E, n, med_p, med_m)
    Graph_E(E, range_n, "n")

    E.clear()

    for m in range_m:    # исследование относительно количества эпох
        print("\n\nM = ", m)
        Forecast(E, med_n, med_p, m)
    Graph_E(E, range_m, "M")

    Graph(Forecast(E, med_n, med_p, 30000), med_p)
