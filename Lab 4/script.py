# ЛР № 4
# Исследование нейронных сетей с радиальными базисными функциями
# (RBF) на примере моделирования булевых выражений

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

n = 0.3

def F(x): # возвращает результат моделируемой булевой функции
    return int(x[2] and x[3] or not(x[0]) or not(x[1]))

def Y(net): # возвращает результат пороговой ФА
    return 1 if net >= 0 else 0

def DeltaW(x, q): # находит величину, на которую изменятся Wi, для пороговой ФА
    return n * q * x

def Net(x, w): # находит значение сетевого входа НС
    return sum([w_i * x_i for w_i, x_i in zip(w[1:], x)]) + w[0]


def Fi(x, c):
    return np.exp((-1) * sum([ (x_i - c_i) ** 2  for x_i, c_i in zip(x, c)]))

def FindC(X):
    RightF = [F(x_i) for x_i in X]
    count_0 = RightF.count(0)
    if (count_0 <= len(RightF)/2) :
        index = [i for i, e in enumerate(RightF) if e == 0]
    else:
        index = [i for i, e in enumerate(RightF) if e == 1]
    return [X[i] for i in index]

def FindFi(X, C):
    fi = [[Fi(X[i], C[j]) for j in range(len(C))] for i in range(len(X))]
    return fi

def MinimazeSet(X): # находит минимальные наборы из общей выборки, на которых возможно обучение НС
    RightF = [F(x_i) for x_i in X]
    TryF = [0 for i in range(len(X))]
    for min_num in range(0, len(X) + 1):
        for min_x in list(combinations(X, min_num)):
            C = FindC(min_x)
            if (len(C) == 0): continue
            Q = 0
            fi = FindFi(min_x, C)
            E, w = RBF(min_x, fi)
            fi = FindFi(X, C)
            for i in range(len(X)):
                TryF[i] = Y(Net(fi[i], w))
                Q += (RightF[i] - TryF[i]) ** 2
            if(Q == 0):
                return E
    return []

def RBF(X, fi): #производит обучение НС и возвращает вектор ошибок Е(к)
                            # и вектор синаптических коэффициентов, на которых обучилась НС
    print("X :", X)
    RightF = [F(x_i) for x_i in X]
    w = [0 for i in range(len(fi[0]) + 1)]
    TryF = [0 for i in range(len(X))]
    E = [1]
    k = 0
    while E[k] != 0:
        print("\n\nk: ", k)
        print("w: ", np.round(w, 3))
        E.append(0)
        for i in range(len(X)):
            net = Net(fi[i], w)
            TryF[i] = Y(net)
            q = RightF[i] - TryF[i]
            for j in range(len(fi[i])):
                w[j + 1] += DeltaW(fi[i][j], q)
            w[0] += DeltaW(1, q)
            E[k+1] += q ** 2
        print("TryF: ", TryF)
        print("E: ", E[k+1])
        k += 1
    return E[1:], w

def Graph(E, name): # строит график зависимости вектора ошибок от эпохи
    if(len(E) == 0): return
    plt.plot(E, 'go-', linewidth=3, markersize=7)
    plt.grid(True)
    plt.title(name)
    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.show()


if __name__=="__main__":

    X = np.unpackbits(np.array([[j] for j in range(2 ** 4)], dtype=np.uint8), axis=1)[:, 4:]

    C = FindC(X)
    fi = FindFi(X, C)

    print("Пороговая ФА")
    E, w = RBF(X, fi)
    Graph(E, "Пороговая ФА")

    print("Пороговая ФА  на минимальных наборах")
    E = MinimazeSet(X)
    Graph(E, "Пороговая ФА на минимальных наборах")
