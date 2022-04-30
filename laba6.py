import numpy as np
import scipy.linalg as spl
from typing import Tuple, List
import random
from math import sqrt
import itertools
from enum import Enum
from prettytable import PrettyTable

def get_circle_radius(matrix: np.ndarray, row_index: int, squared: bool = False) -> float:
    if squared:
        return (matrix[row_index] ** 2).sum() - (matrix[row_index][row_index] ** 2)

    return abs(matrix[row_index]).sum() - abs(matrix[row_index][row_index])


def get_gershgorin_circles(matrix: np.ndarray) -> List[Tuple[float, float]]:
    circles = []
    for i in range(matrix.shape[0]):
        radius = get_circle_radius(matrix, i)
        center = matrix[i][i]
        circles.append((center - radius, center + radius))
    return circles


def rotate(matrix: np.ndarray, index: Tuple[int, int]) -> None:
    i, j = index

    matrix_ii, matrix_ij, matrix_jj = matrix[i][i], matrix[i][j], matrix[j][j]

    x = -2 * matrix_ij
    y = matrix_ii - matrix_jj

    if y == 0:
        cos_phi = sqrt(2) / 2
        sin_phi = sqrt(2) / 2
    else:
        cos_phi = sqrt(1 / 2 * (1 + abs(y) / sqrt(x ** 2 + y ** 2)))
        sin_phi = np.sign(x * y) * abs(x) / (2 * cos_phi * sqrt(x ** 2 + y ** 2))

    matrix[i][i] = cos_phi ** 2 * matrix_ii - 2 * sin_phi * cos_phi * matrix_ij + sin_phi ** 2 * matrix_jj
    matrix[j][j] = sin_phi ** 2 * matrix_ii + 2 * sin_phi * cos_phi * matrix_ij + cos_phi ** 2 * matrix_jj

    matrix[i][j] = (cos_phi ** 2 - sin_phi ** 2) * matrix_ij + sin_phi * cos_phi * (matrix_ii - matrix_jj)
    matrix[j][i] = (cos_phi ** 2 - sin_phi ** 2) * matrix_ij + sin_phi * cos_phi * (matrix_ii - matrix_jj)

    for k in range(matrix.shape[0]):
        if k != i and k != j:
            matrix_ik, matrix_jk = matrix[i][k], matrix[j][k]

            matrix[i][k] = cos_phi * matrix_ik - sin_phi * matrix_jk
            matrix[k][i] = cos_phi * matrix_ik - sin_phi * matrix_jk

            matrix[j][k] = sin_phi * matrix_ik + cos_phi * matrix_jk
            matrix[k][j] = sin_phi * matrix_ik + cos_phi * matrix_jk


def find_abs_max_index(matrix: np.ndarray) -> Tuple[int, int]:
    matrix = matrix - np.diag(np.diag(matrix))
    return np.unravel_index(np.argmax(abs(matrix)), matrix.shape)


def get_line_by_line_indices(shape: Tuple[int, int]):
    indicies = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i != j:
                indicies.append((i, j))
    return itertools.cycle(indicies)


def get_vector_of_square_sums(matrix: np.ndarray) -> np.ndarray:
    return np.array([get_circle_radius(matrix, i, squared=True) for i in range(matrix.shape[0])])


def get_pivot_with_max_radius(matrix: np.ndarray, vector: np.ndarray) -> Tuple[int, int]:
    i = vector.argmax()
        
    matrix = matrix - np.diag(np.diag(matrix))
    j = abs(matrix[i]).argmax()

    return i, j


class PivotStrategy(Enum):
    ABS_MAX = 'abs_max'
    LINE_BY_LINE = 'line_by_line'
    MAX_RADIUS = 'max_radius'


def find_eigen_values(
        matrix: np.ndarray,
        *,
        eps: float,
        strategy: PivotStrategy,
        limit: int = 10000,
) -> Tuple[np.ndarray, int]:
    matrix = np.copy(matrix)

    circle_indicies = get_line_by_line_indices(matrix.shape)
    square_sums_vector = get_vector_of_square_sums(matrix)

    for step in range(limit):
        if strategy is PivotStrategy.MAX_RADIUS:
            if all(get_circle_radius(matrix, i, squared=True) < eps ** 2 for i in range(matrix.shape[0])):
                return np.diag(matrix), step
        else:
            if all(get_circle_radius(matrix, i) < eps for i in range(matrix.shape[0])):
                return np.diag(matrix), step

        if strategy is PivotStrategy.MAX_RADIUS:
            index = get_pivot_with_max_radius(matrix, square_sums_vector)
            square_sums_vector[index[0]] = get_circle_radius(matrix, index[0], squared=True)
            square_sums_vector[index[1]] = get_circle_radius(matrix, index[1], squared=True)
            rotate(matrix, index)
        elif strategy is PivotStrategy.LINE_BY_LINE:
            index = next(circle_indicies)
            rotate(matrix, index)
        elif strategy is PivotStrategy.ABS_MAX:
            index = find_abs_max_index(matrix)
            rotate(matrix, index)

    return np.diag(matrix), step + 1


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

    table_abs_max = PrettyTable()
    table_line_by_line = PrettyTable()
    table_max_radius = PrettyTable()

    table_abs_max.field_names = ["Эпсилон", "Количтество итераций стратегии максимальный по модулю", "Ошибка стратегии максимальный по модулю"]
    table_line_by_line.field_names = ["Эпсилон", "Количтество итераций стратегии циклический выбор", "Ошибка стратегии циклический выбор"]
    table_max_radius.field_names = ["Эпсилон", "Количтество итераций стратегии максимального радиуса", "Ошибка стратегии максимального радиуса"]

    circles = get_gershgorin_circles(A)

    print("Области в которые должны попадать с. ч. матрицы:")
    print(circles)
    print()

    print("Собственные числа матрицы:")
    print(eighs)
    print()
    print()

    eigens, count_abs_maxs = find_eigen_values(A, eps=1e-12, strategy=PivotStrategy.ABS_MAX, limit=2000)
    print(eigens)


    for order in range(-12, -2):
        eps = 10 ** order

        eigen_abs_max, count_abs_max = find_eigen_values(A, eps=eps, strategy=PivotStrategy.ABS_MAX, limit=2000)
        error_abs_max = spl.norm(abs(np.sort(eighs)) - abs(np.sort(eigen_abs_max)))
        table_abs_max.add_row([eps, count_abs_max, error_abs_max])

        eigen_line_by_line, count_line_by_line = find_eigen_values(A, eps=eps, strategy=PivotStrategy.LINE_BY_LINE, limit=2000)
        error_line_by_line = spl.norm(abs(np.sort(eighs)) - abs(np.sort(eigen_line_by_line)))
        table_line_by_line.add_row([eps, count_line_by_line, error_line_by_line])


        eigen_max_radius, count_max_radius = find_eigen_values(A, eps=eps, strategy=PivotStrategy.MAX_RADIUS, limit=2000)
        error_max_radius = spl.norm(abs(np.sort(eighs)) - abs(np.sort(eigen_max_radius)))
        table_max_radius.add_row([eps, count_max_radius, error_max_radius])


    print(table_abs_max)
    print(table_line_by_line)
    print(table_max_radius)



if __name__ == "__main__":
    main()