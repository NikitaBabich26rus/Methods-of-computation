import numpy as np
import scipy.linalg as spl
import numpy.linalg as npl
import random
import math


def solve_L(L, rhs):
    r = rhs.copy()
    s = np.zeros(len(r))
    for i in range(len(r)):
        s[i] = r[i] / L[i, i]
        r[i:] -= L[i:, i] * s[i]
    return s

def solve_U(U, rhs):
    r = rhs.copy()
    s = np.zeros(len(r))
    for i in reversed(range(len(r))):
        s[i] = r[i] / U[i, i]
        r[:i] -= U[:i, i] * s[i]
    return s

def get_T(i, j, zi, zj, ord):
    ans = np.identity(ord)
    invSqrt = 1 / math.sqrt(zi**2 + zj**2)
    ans[i, i] = zi * invSqrt
    ans[j, j] = zi * invSqrt
    ans[i, j] = zj * invSqrt
    ans[j, i] = -zj * invSqrt

    return ans

def get_QR(m):
    n = len(m)
    Qinv = np.identity(n)
    R = m.copy()

    for i in range(n):
        for j in range(i + 1, n):
            Tij = get_T(i, j, R[i, i], R[j, i], n)
            Qinv = Tij @ Qinv
            R = Tij @ R
    return Qinv.T, R


def get_LU(m):
    U = m.copy()
    L = np.identity(len(m))
    for i in range(len(m)):
        coefs = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = coefs
        U[i + 1:] -= coefs[:, np.newaxis] * U[i]
    return L, U


def getM():
    print('Выберите тип матрицы:')
    print('1. Матрица Гильберта')
    print('2. Трёхдиагональная матрица')
    print('3. Случайная')
    opt = int(input('Ваш выбор: '))

    order = int(input('Введите размерность матрицы: '))
    if opt == 1:
        return spl.hilbert(order)
    elif opt == 2:
        return tridiag(order)
    elif opt == 3:
        return np.random.rand(order, order)
    else:
        raise Exception('Ошибка ввода!') 


def tridiag(ord):
    if ord == 1:
        return np.array([[ random.random() ]])

    ans = 5 * np.identity(ord)
    ans[0, 0] = random.random()
    for i in range(ord - 1):
        ans[i, i + 1] = random.random()
        ans[i + 1, i] = random.random()
        ans[i + 1, i + 1] = random.random()

    return ans

def getX(ord):
    print('Использовать случайный вектор x0?')
    ans = input('Ваш выбор (y/n): ')

    if ans == 'y':
        return np.random.rand(ord)
    else:
        return np.ones(ord)

def err(x, x0):
    return spl.norm(x - x0)


def main():
    A = getM()
    x0 = getX(A.shape[0])
    u = A @ x0
    print('Число обусловленности A =', npl.cond(A))
    L, U = get_LU(A)
    Q, R = get_QR(A)
    print('Число обусловленности L =', npl.cond(L))
    print('Число обусловленности U =', npl.cond(U))
    print('Число обусловленности Q =', npl.cond(Q))
    print('Число обусловленности R =', npl.cond(R))
    y = solve_L(L, u)
    x = solve_U(U, y)
    print('LU ошибка вычисления:', err(x, x0))
    y = Q.T @ u
    x = solve_U(R, y)
    print('QR ошибка вычисления:', err(x, x0))

    print('Регуляризация:')
    print('Альфа', 'Ошибка', sep='\t')
    for alphaOrd in range(-12, 0):
        varA = A + (10 ** alphaOrd) * np.identity(A.shape[0])
        print(10 ** alphaOrd, err(npl.solve(varA, u), x0), sep='\t')

if __name__ == "__main__":
    main()
