import numpy as np
import scipy.linalg as spl
import numpy.linalg as npl
import random
import math
from prettytable import PrettyTable


def eighDegree(xc, xp):
    return abs(xc[0] / xp[0])

iterationsMax = 20000
normMax = 1000000000



def degreeMethod(A, eps):
    n = A.shape[0]
    xp = np.ones(n)
    xc = A @ xp
    eigh = eighDegree(xc, xp)
    for i in range(iterationsMax):
        xp = xc
        xc = A @ xp

        if abs(eigh - eighDegree(xc, xp)) < eps:
            break
        eigh = eighDegree(xc, xp)

        if spl.norm(xc) >= normMax:
            xc = xc / spl.norm(xc)
        elif spl.norm(xc) <= 1 / normMax:
            xc = xc / spl.norm(xc)

    return eigh, i



def eighScalar(xc, xp, yc):
    return np.dot(xc, yc) / np.dot(xp, yc)



def scalarMethod(A, eps):
    n = A.shape[0]
    xp = np.ones(n)
    xc = A @ xp

    yp = np.ones(n)
    yc  = A.T @ yp

    eigh = eighScalar(xc, xp, yc)
    for i in range(iterationsMax):
        xp = xc
        xc = A @ xp

        yp = yc
        yc = A.T @ yp

        if abs(eigh - eighScalar(xc, xp, yc)) < eps:
            break
        eigh = eighScalar(xc, xp, yc)

        if spl.norm(xc) >= normMax:
            xc = xc / spl.norm(xc)
            yc = xc / spl.norm(yc)

    return eigh, i



def tridiag(n):
    if n == 1:
        return np.array([[ random.random() ]])

    ans = 5 * np.identity(n)
    ans[0, 0] = random.random()
    for i in range(n - 1):
        ans[i, i + 1] += random.random()
        ans[i + 1, i] = ans[i, i + 1]
        ans[i + 1, i + 1] += random.random()

    return ans



def getM():
    print('Выберите тип матрицы:')
    print('1. Матрица Гильберта')
    print('2. Трёхдиагональная матрица')
    print('3. Случайная')
    opt = int(input('Ваш выбор: '))

    ns = [int(x) for x in input('Введите порядок матрицы: ').split(" ")]
    for n in ns:
        if n < 1:
            raise Exception('Ошибка')

    if opt == 1:
        return spl.hilbert(n)
    elif opt == 2:
        return tridiag(n)
    elif opt == 3:
        return np.random.rand(n, n) + np.identity(n)
    else:
        raise Exception('Ошибка')



def main():
    A = getM()
    eighs, eighvec = spl.eigh(A)
    print('Собственные числа:', eighs)
    print()
    print('Собственные вектора матрицы:', eighvec)
    print()

    for i in range(len(eighs)):
        eighs[i] = abs(eighs[i])

    maxEigh = max(eighs)

    table = PrettyTable()
    table.field_names = ["Эпсилон", "Количтество итераций степенного метода", "Ошибка степенного метода", "Количтество итераций скалярного метода", "Ошибка скалярного метода"]

    for order in range(-10, -2):
        eps = 10 ** order
        degEigh, degIters = degreeMethod(A, eps)
        scalEigh, scalIters = scalarMethod(A, eps)
        table.add_row([eps, degIters, abs(maxEigh - degEigh), scalIters, abs(maxEigh - scalEigh)])

    print(table)


if __name__ == "__main__":
    main()
