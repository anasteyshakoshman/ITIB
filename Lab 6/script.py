

# ЛР № 6
# Изучения алгоритма обратного распространения 
# ошибки ( метод Back Propagation ) 

import matplotlib.pyplot as plt
import random

X1 = [1., 1., -1.]  # входной вектор
T = [0.2, -0.1]  # целевой вектор
n = 0.3


def Q_error2(Y, m):
    return (T[m] - Y[m]) * (1 - Y[m] ** 2) / 2

def Q_error1(w2, q2, out2):
    return sum([w_i * q_i for w_i, q_i in zip(w2, q2)]) * (1 - out2 ** 2) / 2

def DeltaW(x, q):  # находит величину, на которую изменятся Wi
    return n * x * q

def Net(x, w):  # находит значение сетевого входа НС
    return sum([w_i * x_i for w_i, x_i in zip(w[1:], x)]) + w[0]

def Out(net):   # находит выходной сигнал
    return (1 - np.exp((-1) * net)) / (1 + np.exp((-1) * net))

def MeanSquareError(T, Y):
    summa = 0
    for t_i, y_i in zip(T, Y):
        summa += (t_i - y_i) ** 2
    return summa ** 0.5

def Learning(N, J, M):  # обучение НС

    X2 = [0. for j in range(J)]
    Y = [0. for m in range(M)]

    w1 = np.array([[random.uniform(0., 0.9) for i in range(N+1)]  for j in range(J)])
    w2 = np.array([[0.] * (J+1) for m in range(M)])

    q1 = [0. for j in range(J)]
    q2 = [0. for m in range(M)]

    net1 = [0. for j in range(J)]
    net2 = [0. for m in range(M)]

    E = [MeanSquareError(T, Y)]
    era = 0

    while (E[len(E) - 1] > 10 ** (-3)):

        print("\nera = ", np.round(era, 3))

        # первый этап для скрытого слоя
        for j in range(J):
            net1[j] = Net(X1, w1[j])
            X2[j] = Out(net1[j])

        # первый этап для выходного слоя
        for m in range(M):
            net2[m] = Net(X2, w2[m])
            Y[m] = Out(net2[m])

        # второй этап для выходного слоя
        for m in range(M):
            q2[m] = Q_error2(Y, m)

        # второй этап для скрытого слоя
        for j in range(J):
            q1[j] = Q_error1(q2, w2[:, j+1], X2[j])

        # третий этап для скрытого слоя
            for i in range(N):
                w1[j][i] += DeltaW(X1[i], q1[j])

        # третий этап для выходного слоя
        for m in range(M):
            for j in range(J):
                w2[m][j + 1] += DeltaW(X2[j], q2[m])
            w2[m][0] += DeltaW(1, q2[m])

        E.append(MeanSquareError(T, Y))
        era += 1

        print("w1 : \n", np.round(w1, 4))
        print("w2 : \n", np.round(w2, 4))
        print("Y : ", np.around(Y, 3))
        print("e = ", np.round(E[len(E) - 1], 3))

    return E[1:]


def Graph(E):  # строит график зависимости ошибки
    plt.plot(E, 'bo-', linewidth=1, markersize=2)
    plt.title("E(k)")
    plt.xlabel("k")
    plt.ylabel('E')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    E = Learning(2,1,2)
    Graph(E)
