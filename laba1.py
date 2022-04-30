import numpy as np
import scipy as sp
import scipy.linalg as spl
import random


def tridiag(order):
    if order == 1:
        return [[ random.random() ]]

    answer = 5 * np.identity(order)
    answer[0, 0] = random.random()
    for i in range(order - 1):
        answer[i, i + 1] += random.random()
        answer[i + 1, i] += random.random()
        answer[i + 1, i + 1] += random.random()

    return answer


def volumetric_cond(m):
    answer = 1
    for row in m:
        answer *= np.linalg.norm(row)
    return answer / abs(np.linalg.det(m))


def angular_cond(m):
    answer = 0
    invert_matrix = np.linalg.inv(m)
    for (row, row_inv) in zip(m, invert_matrix):
        answer = max(answer, np.linalg.norm(row) * np.linalg.norm(row_inv))

    return answer


def get_matrix():
    print('Выберите тип матрицы:')
    print('1. Матрица Гильберта')
    print('2. Трёхдиагональная матрица')
    print('3. Случайная')
    choise = int(input('Ваш выбор: '))

    order = int(input('Введите размерность матрицы: '))
    if choise == 1:
        return spl.hilbert(order)
    elif choise == 2:
        return tridiag(order)
    elif choise == 3:
        return np.random.rand(order, order)
    else:
        raise Exception('Ошибка ввода!')    

def calc_error(current_answer):
    return np.linalg.norm(current_answer - np.ones(len(current_answer)))
    

def main():
    matrix = get_matrix()
    u = matrix @ np.ones(len(matrix))
    print('Спектральный критерий обусловленности:  ', np.linalg.cond(matrix))
    print('Объемный критерий обусловленности:', volumetric_cond(matrix))
    print('Угловой критерий обусловленности:   ', angular_cond(matrix))

    print('\nОкругление\tОшибка')
    for varOrd in range(-10, -1):
        uRound = np.round(u, -varOrd)
        matrixRound = np.round(matrix, -varOrd)
        answer = np.linalg.solve(matrixRound, uRound)
        print(varOrd, calc_error(answer), sep='\t\t')
        


if __name__ == "__main__":
    main()
