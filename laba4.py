import numpy as np
import scipy.linalg as spl
import numpy.linalg as npl
import random
from prettytable import PrettyTable
import math

iterationsMax = 20000
normMax = 1000000000

def solveSimple(A, b, eps):
    n = A.shape[0]
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            B[i, j] = -A[i, j] / A[i, i]
            B[j, i] = -A[j, i] / A[j, j]

    c = b.copy()
    for i in range(n):
        c[i] /= A[i, i]

    xc = np.zeros(n)
    for i in range(iterationsMax + 1):
        xp = xc
        xc = B @ xp + c

        if spl.norm(xc) > normMax:
            return iterationsMax
        if spl.norm(xc - xp) < eps:
            break

    return i
    

def zeidelNext(L, D, R, b, xp):
    return -npl.inv(D + L) @ R @ xp + npl.inv(D + L) @ b

def solveZeidel(A, b, eps):
    n = A.shape[0]

    D = np.diag(np.diagonal(A))
    L = np.tril(A) - D
    R = np.triu(A) - D

    xc = np.zeros(n)
    for i in range(iterationsMax + 1):
        xp = xc
        xc = zeidelNext(L, D, R, b, xp)

        if spl.norm(xc) > normMax:
            return iterationsMax
        if spl.norm(xc - xp) < eps:
            break

    return i


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
    n = A.shape[0]
    b = A @ np.ones(n)

    table = PrettyTable()
    table.field_names = ["Эпсилон", "Метод простой итерации", "Метод Зейделя"]

    for order in range(-12, -1):
        eps = 10 ** order
        iSimple = solveSimple(A, b, eps)
        iZeidel = solveZeidel(A, b, eps)

        table.add_row([eps, iSimple, iZeidel])

    print(table)


if __name__ == "__main__":
    main()
