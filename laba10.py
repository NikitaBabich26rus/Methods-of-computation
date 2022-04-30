from math import sin, pi, e, log
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from itertools import product

from functools import partialmethod

Function1 = Callable[[float], float]
Function2 = Callable[[float, float], float]



def solve_linear_system(matrix: np.ndarray, right_side: np.ndarray) -> np.ndarray:
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



def solve_heat_equation(
        kappa: float,
        f: Function2,
        line_limit: float,
        time_limit: float,
        number_of_dimensional_layers: int,
        number_of_time_layers: int,
        init_condition: Function1,
        left_condition: Function1,
        right_condition: Function1,
        sigma: float,
) -> np.ndarray:
    time_grid, tau = np.linspace(0, time_limit, number_of_time_layers, retstep=True)
    dimensional_grid, h = np.linspace(0, line_limit, number_of_dimensional_layers, retstep=True)

    heat_matrix = [
        [init_condition(point) for point in dimensional_grid],
    ]

    for i, time in enumerate(time_grid[1:]):
        matrix = np.zeros((number_of_dimensional_layers, number_of_dimensional_layers))
        right_side = np.empty(number_of_dimensional_layers)

        matrix[0, 0] = 1
        right_side[0] = left_condition(time)

        for j, (left, middle, right) in enumerate(zip(heat_matrix[i], heat_matrix[i][1:], heat_matrix[i][2:]), start=1):
            matrix[j, j - 1] = -1 * tau * kappa * sigma
            matrix[j, j] = h ** 2 + 2 * tau * kappa * sigma
            matrix[j, j + 1] = -1 * tau * kappa * sigma

            right_side[j] = (
                    h ** 2 * middle + (1 - sigma) * (left - 2 * middle + right) * kappa * tau
                    + tau * h ** 2 * f(dimensional_grid[j], (time - tau) + sigma * tau)
            )

        matrix[number_of_dimensional_layers - 1, number_of_dimensional_layers - 1] = 1
        right_side[number_of_dimensional_layers - 1] = right_condition(time)

        heat_matrix.append(solve_linear_system(matrix, right_side))

    return np.array(heat_matrix)



def get_grid_by_solution(
        true_solution: Function2,
        line_limit: float,
        time_limit: float,
        number_of_dimensional_layers: int,
        number_of_time_layers: int,
) -> np.ndarray:
    return np.array([
        [true_solution(x, time) for x in np.linspace(0, line_limit, number_of_dimensional_layers)]
        for time in np.linspace(0, time_limit, number_of_time_layers)
    ])


def _get_error(true_grid: np.ndarray, actual_grid: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    error_grid = abs(true_grid - actual_grid)

    return error_grid.max(), np.unravel_index(error_grid.argmax(), actual_grid.shape)


def compare_grids(
        actual_grid: np.ndarray,
        true_grid: np.ndarray,
        line_limit: float,
        time_limit: float,
) -> None:
    x = np.linspace(0, time_limit, actual_grid.shape[0])
    y = np.linspace(0, line_limit, actual_grid.shape[1])

    fig = go.Figure()

    fig.add_surface(x=y, y=x, z=true_grid)

    yv, xv = np.meshgrid(y, x)
    fig.add_scatter3d(y=xv.flatten(), x=yv.flatten(), z=actual_grid.flatten(), marker=dict(size=2, color="blue"), mode='markers')

    fig.update_layout(
        title='Сравнение истинного решения и найденной сетки',
        scene_zaxis_title='Температура',
        scene_yaxis_title='Время',
        scene_xaxis_title='Координата стержня',
        height=800,
    )

    fig.show()


def check_steps(        
        line_limit: float,
        time_limit: float,
        number_of_dimensional_layers: int,
        number_of_time_layers: int,
        kappa: float,
) -> None:
    h = line_limit / (number_of_dimensional_layers - 1)
    tau = time_limit / (number_of_time_layers - 1)

    print(f'h: {h}')
    print(f'tau: {tau}\n')

    print(f'2 * kappa * tau: {2 * kappa * tau}')
    print(f'h^2: {h ** 2}\n')

    if 2 * kappa * tau <= h ** 2:
        print(f'Условие устойчивости выполняется!\n')
    else:
        print(f'Условие устойчивости не выполняется!\n')


def show_error(true_grid: np.ndarray, actual_grid: np.ndarray,line_limit: float, time_limit: float) -> None:
    error, index = _get_error(true_grid, actual_grid)

    x = np.linspace(0, line_limit, actual_grid.shape[1])
    t = np.linspace(0, time_limit, actual_grid.shape[0])

    print(f'Error: {error}')
    print(f'Bad point: ({index[1]}, {index[0]}) = ({x[index[1]]}, {t[index[0]]})\n')



def main():
    kappa = 1
    f = lambda x, t: 2 * t + sin(x)

    init_condition = lambda x: sin(x)
    left_condition = lambda t: t ** 2
    right_condition = lambda t: t ** 2 - 1

    line_limit = 3 * pi / 2
    time_limit = 5

    true_solution = lambda x, t: sin(x) + t ** 2

    number_of_dimensional_layers = 50
    number_of_time_layers = 50
    sigma = 0

    true_grid = get_grid_by_solution(true_solution, line_limit, time_limit, number_of_dimensional_layers, number_of_time_layers)

    actual_grid = solve_heat_equation(
        kappa=kappa, 
        f=f, 
        line_limit=line_limit, 
        time_limit=time_limit, 
        number_of_dimensional_layers=number_of_dimensional_layers, 
        number_of_time_layers=number_of_time_layers, 
        init_condition=init_condition, 
        left_condition=left_condition, 
        right_condition=right_condition, 
        sigma=sigma,
    )

    show_error(true_grid, actual_grid, line_limit, time_limit)
    check_steps(line_limit, time_limit, number_of_dimensional_layers, number_of_time_layers, kappa)
    compare_grids(actual_grid, true_grid, line_limit, time_limit)


    number_of_dimensional_layers = 30
    number_of_time_layers = 380
    sigma = 0

    true_grid = get_grid_by_solution(true_solution, line_limit, time_limit, number_of_dimensional_layers, number_of_time_layers)

    actual_grid = solve_heat_equation(
        kappa=kappa, 
        f=f, 
        line_limit=line_limit, 
        time_limit=time_limit, 
        number_of_dimensional_layers=number_of_dimensional_layers, 
        number_of_time_layers=number_of_time_layers, 
        init_condition=init_condition, 
        left_condition=left_condition, 
        right_condition=right_condition, 
        sigma=sigma,
    )

    show_error(true_grid, actual_grid, line_limit, time_limit)
    check_steps(line_limit, time_limit, number_of_dimensional_layers, number_of_time_layers, kappa)
    compare_grids(actual_grid, true_grid, line_limit, time_limit)

if __name__ == "__main__":
    main()