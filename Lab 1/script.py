# ЛР № 1
# Исследование однослойных нейронных сетей
# на примере моделирования булевых выражений

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

n = 0.3

def Hemming(f1, f2): # возвращает расстояние Хэмминга между наборами  f1 и f2
    e = 0
    for x1, x2 in zip(f1, f2):
        if(x1 != x2) : e += 1
    return e

def F(x): # возвращает результат моделируемой булевой функции
    return int(x[0] or x[1] or x[2]) and x[3]

def Y1(net): # возвращает результат пороговой ФА
    return 1 if net >= 0 else 0

def Y2(net): # возвращает результат логистической ФА
    out = 0.5 * (np.tanh(net) + 1)
    return 1 if out >= 0.5 else 0

def DeltaW1(x, q, net): # находит величину, на которую изменятся Wi, для пороговой ФА
    return n * q * x

def DeltaW2(x, q, net): # находит величину, на которую изменятся Wi, для логистической ФА
    return n * q * x * ((-0.5) * (np.tanh(net) ** 2) + 0.5)

def Net(x, w): # находит значение сетевого входа НС
    return sum([w_i * x_i for w_i, x_i in zip(w[1:], x)]) + w[0]

def LearningNN(X, Y, DeltaW): #производит обучение НС и возвращает вектор ошибок Е(к)
                       # и вектор синаптических коэффициентов, на которых обучилась НС
    RightF = [F(x_i) for x_i in X]
    w = [0 for i in range(5)]
    TryF = [0 for i in range(len(X))]
    E = [1]
    k = 0

    while E[k]:
        print("\n\nk =", k, "\n")
        print("w =", w)
        E.append(0)
        for i in range(len(X)):
            net = Net(X[i], w)
            TryF[i] = Y(net)
            q = RightF[i] - TryF[i]
            for j in range(len(X[i])):
                w[j + 1] += DeltaW(X[i][j], q, net)
            w[0] += DeltaW(1, q, net)
            E[k+1] += q ** 2
        k += 1
        print( TryF)
        print("\nE : ", E[k])


    return E, w

def Graph(E, name): # строит график зависимости вектора ошибок от эпохи
    plt.plot(E[1:], 'go-', linewidth = 3, markersize = 7 )
    plt.grid(True)
    plt.title(name)
    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.show()

def MinimazeSet(X, Y, DeltaW, name): # находит минимальные наборы из общей выборки, на которых возможно обучение НС
    RightF = [F(x_i) for x_i in X]
    TryF = [0 for i in range(0, len(X))]
    for min_num in range(0, len(X) + 1):
        for min_x in list(combinations(X, min_num)):
            E, w = LearningNN(min_x, Y, DeltaW)
            for i in range(0, len(X)):
                net = Net(X[i], w)
                TryF[i] = Y2(net)
            if(Hemming(RightF, TryF) == 0) :
                Graph(E, name + "(c минимальным кол-вом наборов)")
                return min_x, w


if __name__=="__main__":

    X = np.unpackbits(np.array([[j] for j in range(0, pow(2, 4))], dtype=np.uint8), axis=1)[:, 4:]

    E, w = LearningNN(X, Y1, DeltaW1)
    Graph(E, "Пороговая ФА")

    E, w = LearningNN(X, Y2, DeltaW2)
    Graph(E, "Логистическая ФА")

    min_x, w = MinimazeSet(X, Y1, DeltaW1, "Пороговая ФА")
    print("\n\nmin_x :", min_x, "\n\nw =", w)

    min_x, w = MinimazeSet(X, Y2, DeltaW2, "Логистическая ФА")
    print("\n\nmin_x :", min_x, "\n\nw =", w)

