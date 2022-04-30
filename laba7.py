from collections import defaultdict
from distutils.log import error
from math import sin, cos, e, log
from timeit import default_timer as timer
from typing import Callable, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse import lil_matrix, vstack, coo_matrix
from scipy.sparse.linalg import spsolve
from prettytable import PrettyTable
import plotly.express as px

from functools import partialmethod


Function = Callable[[float], float]
Grid = Tuple[np.ndarray, float]
GeneralCondition = Tuple[float, float, float]



def solve(
        matrix: lil_matrix,
        right_side: np.ndarray,
        matrix_type: Literal['tridiagonal', 'other'] = 'other',
) -> np.ndarray:
    if matrix_type == 'other':
        return spsolve(matrix, right_side)

    alphas = []
    betas = []

    # Прямой ход: находим коэффициенты
    for i in range(matrix.shape[0]):
        y = matrix[i, i] if i == 0 else matrix[i, i] + matrix[i, i - 1] * alphas[i - 1]
        alphas.append(0 if i == (matrix.shape[0] - 1) else -1 * matrix[i, i + 1] / y)
        betas.append(right_side[i] / y if i == 0 else (right_side[i] - matrix[i, i - 1] * betas[i - 1]) / y)

    # Обратный ход: находим x
    reversed_solution = []
    for i in reversed(range(matrix.shape[0])):
        reversed_solution.append(
            betas[i]
            if i == (matrix.shape[0] - 1) else
            alphas[i] * reversed_solution[matrix.shape[0] - i - 2] + betas[i]
        )
    return np.array(reversed_solution[::-1])



def solve_equation_using_finite_difference_method(
        q: Function,
        r: Function,
        f: Function,
        grid: Grid,
        left_general_condition: GeneralCondition,
        right_general_condition: GeneralCondition,
) -> np.ndarray:
    left_coefficient_1, left_coefficient_2, left_value = left_general_condition
    right_coefficient_1, right_coefficient_2, right_value = right_general_condition

    matrix_type = 'tridiagonal' if left_coefficient_2 == 0 and right_coefficient_2 == 0 else 'other'

    points, step = grid

    first_row = [
        left_coefficient_1 + 3 / (2 * step) * left_coefficient_2,
        -2 * left_coefficient_2 / step,
        left_coefficient_2 / (2 * step),
    ]
    first_row.extend([0] * (len(points) - 3))

    matrix = coo_matrix([first_row])
    right_side = [left_value]

    for index, point in enumerate(points[1:-1]):
        row = [0] * index

        row.append(1 / (step ** 2) - q(point) / (2 * step))
        row.append(-2 / (step ** 2) - r(point))
        row.append(1 / (step ** 2) + q(point) / (2 * step))

        row.extend([0] * (len(points) - index - 3))

        matrix = vstack([matrix, row])
        right_side.append(f(point))

    last_row = [float(0)] * (len(points) - 3)
    last_row.extend([
        right_coefficient_2 / (2 * step),
        -2 * right_coefficient_2 / step,
        right_coefficient_1 + 3 * right_coefficient_2 / (2 * step),
    ])

    matrix = vstack([matrix, last_row], format='lil')
    right_side.append(right_value)

    return solve(matrix, np.array(right_side), matrix_type)

def _get_richardson_error(curr_solution: np.ndarray, next_solution: np.ndarray) -> float:
    error_by_even_points = abs(next_solution[::2] - curr_solution) / 3

    max_error = -1
    for left_error, right_error in zip(error_by_even_points, error_by_even_points[1:]):
        max_error = max(max_error, left_error)
        max_error = max(max_error, (left_error + right_error) / 2)
        max_error = max(max_error, right_error)

    return max_error



def _get_true_error(true_solution: np.ndarray, actual_solution: np.ndarray) -> float:
    return abs(true_solution - actual_solution).max()


def get_result(q: Function,
        r: Function,
        f: Function,
        a: float,
        b: float,
        left_condition: GeneralCondition,
        right_condition: GeneralCondition,
        true_function: Optional[Callable[[float], float]] = None,
        eps: Optional[float] = None,
        limit: int = 5,
    ):

    true_function = np.vectorize(true_function)
    table = PrettyTable()
    table.field_names = ["Эпсилон", "Число разбиений", "Ошибка Ричардсона", "Настоящая ошибка"]


    number_of_points = 4
    current_grid = np.linspace(a, b, number_of_points, retstep=True)
    current_result = solve_equation_using_finite_difference_method(q=q, r=r, f=f, grid=current_grid, left_general_condition=left_condition, right_general_condition=right_condition)

    for i in range(1, limit):

        number_of_points = number_of_points * 2 - 1
        current_grid = np.linspace(a, b, number_of_points, retstep=True)

        result = solve_equation_using_finite_difference_method(q=q, r=r, f=f, grid=current_grid, left_general_condition=left_condition, right_general_condition=right_condition)

        richardson_error = _get_richardson_error(current_result, result)

        true_error = _get_true_error(true_function(current_grid[0]), result)
        current_result = result

        table.add_row([eps, number_of_points, richardson_error, true_error])

        if (richardson_error <= eps):
            print(table)
            return

    print(table)



def main():
    u = lambda x: x * e ** sin(x)
    du = lambda x: e ** sin(x) * (x * cos(x) + 1)
    ddu = lambda x: e ** sin(x) * (-x * sin(x) + x * cos(x) ** 2 + 2 * cos(x))

    q = lambda x: -x
    r = lambda x: 1
    f = lambda x: ddu(x) + q(x) * du(x) - r(x) * u(x)

    a = -2
    b = 6

    alpha_1 = 1
    alpha_2 = 0
    alpha = -2 * e ** sin(-2)

    left_condition = (alpha_1, alpha_2, alpha)

    beta_1 = 1
    beta_2 = 0
    beta = 6 * e ** sin(6)

    right_condition = (beta_1, beta_2, beta)

    get_result(q=q,
        r=r,
        f=f,
        a=a,
        b=b,
        left_condition=left_condition,
        right_condition=right_condition,
        true_function=u,
        eps=1e-10,
        limit=13
    )

    u = lambda x: x ** 5 - 5 * x ** 4 + 5 * x ** 3 + 5 * x ** 2 - 6 * x
    du = lambda x: 5 * x ** 4 - 20 * x ** 3 + 15 * x ** 2 + 10 * x - 6
    ddu = lambda x: 10 * (2 * x ** 3 - 6 * x ** 2 + 3 * x + 1)

    q = lambda x: 5
    r = lambda x: -x
    f = lambda x: ddu(x) + q(x) * du(x) - r(x) * u(x)

    a = -1
    b = 3

    alpha_1 = 2
    alpha_2 = 1
    alpha = -24

    left_condition = (alpha_1, alpha_2, alpha)

    beta_1 = 3
    beta_2 = 1 / 8
    beta = 3

    right_condition = (beta_1, beta_2, beta)

    get_result(q=q,
        r=r,
        f=f,
        a=a,
        b=b,
        left_condition=left_condition,
        right_condition=right_condition,
        true_function=u,
        eps=1e-10,
        limit=13
    )

if __name__ == "__main__":
    main()