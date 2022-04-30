from dataclasses import dataclass
from math import e, sin, cos, log, pi
from typing import List, Tuple, Callable, Optional

import numpy as np
import numpy.linalg as linalg
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import quad
from scipy.special import eval_jacobi



Function = Callable[[np.float64], np.float64]
RESOLUTION = 15000


@dataclass
class GeneralCondition:
    """
    Граничное условие общего вида: alpha_1 * u(a) ± alpha_2 * u'(a) = alpha, где

    alpha_1 = first_coefficient,
    alpha_2 = second_coefficient,
    alpha = expected_value,
    a = point
    """
    expected_value: float
    point: float

    first_coefficient: float = 1
    second_coefficient: float = 0

def get_jacobi_polynomials_with_derivative(
        *,
        n: int,
        k: int,
) -> Tuple[Function, Function]:
    polynomial: Function = lambda x: eval_jacobi(n, k, k, x)

    if n == 0:
        derivative: Function = lambda x: 0
    else:
        derivative: Function = lambda x: (n + 2 * k + 1) / 2 * eval_jacobi(n - 1, k + 1, k + 1, x)

    return polynomial, derivative


def get_basis_functions_with_derivatives(
        *,
        number_of_basis_functions: int,
        left_condition: GeneralCondition,
        right_condition: GeneralCondition,
        k_jacobi_polynomials: int = 2,
) -> Tuple[List[Function], List[Function]]:
    basis_functions = []
    basis_derivatives = []

    for i in range(number_of_basis_functions):
        basis_function, basis_derivative = get_basis_function_with_derivative(
            number_of_basis_function=i,
            left_condition=left_condition,
            right_condition=right_condition,
            k_jacobi_polynomials=k_jacobi_polynomials,
        )
            
        basis_functions.append(basis_function)
        basis_derivatives.append(basis_derivative)

    return basis_functions, basis_derivatives




def get_basis_function_with_derivative(
        *,
        number_of_basis_function: int,
        left_condition: GeneralCondition,
        right_condition: GeneralCondition,
        k_jacobi_polynomials: int = 2,
) -> Tuple[Function, Function]:
    alpha_1 = left_condition.first_coefficient
    alpha_2 = left_condition.second_coefficient

    beta_1 = right_condition.first_coefficient
    beta_2 = right_condition.second_coefficient

    a = left_condition.point
    b = right_condition.point

    matrix = np.array([
        [alpha_1 * a - alpha_2, alpha_1],
        [beta_1 * b + beta_2, beta_1],
    ])

    if number_of_basis_function == 0:
        right_side = np.array([
            [a * (-1 * alpha_1 * a + 2 * alpha_2)],
            [b * (-1 * beta_1 * b - 2 * beta_2)],
        ])

        c_1, d_1 = linalg.solve(matrix, right_side)

        return np.vectorize(lambda x: x ** 2 + c_1 * x + d_1), np.vectorize(lambda x: 2 * x + c_1)
    elif number_of_basis_function == 1:
        right_side = np.array([
            [a ** 2 * (-1 * alpha_1 * a + 3 * alpha_2)],
            [b ** 2 * (-1 * beta_1 * b - 3 * beta_2)],
        ])

        c_2, d_2 = linalg.solve(matrix, right_side)

        return np.vectorize(lambda x: x ** 3 + c_2 * x + d_2), np.vectorize(lambda x: 3 * x ** 2 + c_2)
    else:
        polynomial, derivative = get_jacobi_polynomials_with_derivative(n=number_of_basis_function - 2, k=k_jacobi_polynomials)
        new_polynomial = lambda x: (x - a) ** 2 * (x - b) ** 2 * polynomial((2 * x - b - a) / (b - a))
        new_derivative = lambda x: (
                (x - a) ** 2 * (x - b) ** 2 * derivative((2 * x - b - a) / (b - a)) * (2 / (b - a))
                + 2 * polynomial((2 * x - b - a) / (b - a)) * (x - a) * (x - b) * (2 * x - a - b)
        )
        return new_polynomial, new_derivative




def solve_using_galerkin_method(
        *,
        k: Function,
        v: Function,
        q: Function,
        f: Function,
        left_condition: GeneralCondition,
        right_condition: GeneralCondition,
        k_derivative: Function,
        number_of_basis_functions: int,
) -> Function:
    alpha_1 = left_condition.first_coefficient
    alpha_2 = left_condition.second_coefficient
    alpha = left_condition.expected_value

    beta_1 = right_condition.first_coefficient
    beta_2 = right_condition.second_coefficient
    beta = right_condition.expected_value

    a = left_condition.point
    b = right_condition.point

    z = lambda x: 0
    new_f = f

    if alpha != 0 or beta != 0:
        matrix = np.array([
            [a * alpha_1 - alpha_2, alpha_1],
            [b * beta_1 + beta_2, beta_1],
        ])
        right_side = np.array([alpha, beta])

        d_1, d_2 = linalg.solve(matrix, right_side)
        z = lambda x: d_1 * x + d_2

        new_f = lambda x: f(x) - (-1 * k_derivative(x) * d_1 + v(x) * d_1 + q(x) * (d_1 * x + d_2))

    basis_functions, basis_derivatives = get_basis_functions_with_derivatives(
        number_of_basis_functions=number_of_basis_functions,
        left_condition=left_condition,
        right_condition=right_condition,
    )

    matrix = []
    right_side = []

    x = np.linspace(a, b)

    for basis_function_i, basis_derivative_i in zip(basis_functions, basis_derivatives):
        matrix_row = []
        for basis_function_j, basis_derivative_j in zip(basis_functions, basis_derivatives):
            matrix_coefficient = quad(
                func=lambda x: (
                        k(x) * basis_derivative_j(x) * basis_derivative_i(x)
                        + v(x) * basis_derivative_j(x) * basis_function_i(x)
                        + q(x) * basis_function_j(x) * basis_function_i(x)
                ),
                a=a,
                b=b,
            )[0]

            matrix_row.append(matrix_coefficient)

        matrix.append(matrix_row)

        right_side.append(quad(lambda x: new_f(x) * basis_function_i(x), a=a, b=b)[0])

    matrix = np.array(matrix)
    right_side = np.array(right_side)

    coefficients = linalg.solve(matrix, right_side)

    homogeneous_solution = lambda x: sum(
        basis_function(x) * coefficient for basis_function, coefficient in zip(basis_functions, coefficients)
    )

    return lambda x: homogeneous_solution(x) + z(x)



def calculate_error(
        actual_solution: Function, 
        true_solutiion: Function, 
        segment: Tuple[float, float], 
) -> float:
    points = np.linspace(*segment, RESOLUTION)

    actual_values = actual_solution(points)
    true_values = true_solutiion(points)

    return abs(actual_values - true_values).max()




def _add_true_solution(fig, true_solution: Function, segment: Tuple[float, float], x_bias: float) -> None:
    a, b = segment

    x, step = np.linspace(a, b, RESOLUTION, retstep=True)
    fig.add_scatter(x=x, y=true_solution(x), name='True', line_color='rgba(0, 0, 0, 0.25)', showlegend=False)

    x = np.linspace(a - x_bias, a, int(RESOLUTION * x_bias))
    fig.add_scatter(x=x, y=true_solution(x), name='True', line_color='rgba(0, 0, 0, 0.25)', showlegend=False)

    x = np.linspace(b, b + x_bias, int(RESOLUTION * x_bias))
    fig.add_scatter(x=x, y=true_solution(x), name='True', line_color='rgba(0, 0, 0, 0.25)', showlegend=False)


def get_result(
    k: Function,
    k_derivative: Function,
    v: Function,
    q: Function,
    f: Function,
    left_condition: GeneralCondition,
    right_condition: GeneralCondition,
    limit: int = 20,
    step: int = 1,
    true_solution: Optional[Function] = None,
    x_bias: float = 1,
):

    segment = (left_condition.point, right_condition.point)
    x = np.linspace(*segment, RESOLUTION)

    min_error = float('inf')
    for i in range(2, limit + 1, step):

        actual_solution = solve_using_galerkin_method(
            k=k,
            v=v,
            q=q,
            f=f,
            left_condition=left_condition,
            right_condition=right_condition,
            k_derivative=k_derivative,
            number_of_basis_functions=i,
        )

        if true_solution is not None:
            current_error = calculate_error(actual_solution, true_solution, (left_condition.point, right_condition.point))
            if current_error < min_error:
                min_error = current_error
                best_solution = actual_solution
                best_index = i
    

    if true_solution is not None:
        best_fig = go.Figure()

        x = np.linspace(*segment, RESOLUTION)
        best_fig.add_scatter(x=x, y=best_solution(x), name=f'Лучшее решение с индексом: ({best_index})')

        _add_true_solution(best_fig, true_solution, segment, x_bias)
        best_fig.update_layout(title='Сравнение истинного решение и наилучшего найденного')

        best_fig.show(config={'scrollZoom': True})


def main():

    u = lambda x: x ** 5 - 5 * x ** 4 + 5 * x ** 3 + 5 * x ** 2 - 6 * x
    du = lambda x: 5 * x ** 4 - 20 * x ** 3 + 15 * x ** 2 + 10 * x - 6
    ddu = lambda x: 10 * (2 * x ** 3 - 6 * x ** 2 + 3 * x + 1)

    k = lambda x: x ** 2
    v = lambda x: 5
    q = lambda x: -x

    k_derivative = lambda x: 2 * x
    f = lambda x: -1 * (k_derivative(x) * du(x) + k(x) * ddu(x)) + v(x) * du(x) + q(x) * u(x)

    left_condition = GeneralCondition(expected_value=0, point=-1)
    right_condition = GeneralCondition(expected_value=0, point=3)

    get_result(k, k_derivative, v, q, f, left_condition, right_condition, true_solution=np.vectorize(u))


    u = lambda x: sin(1 / x)
    du = lambda x: -1 * cos(1 / x) / x ** 2
    ddu = lambda x: (2 * x * cos(1 / x) - sin(1 / x)) / x ** 4

    k = lambda x: x
    v = lambda x: x ** 2
    q = lambda x: x ** 3

    k_derivative = lambda x: 1
    f = lambda x: -1 * (k_derivative(x) * du(x) + k(x) * ddu(x)) + v(x) * du(x) + q(x) * u(x)

    left_condition = GeneralCondition(expected_value=0, point=1 / (8 * pi))
    right_condition = GeneralCondition(expected_value=0, point=1 / (4 * pi))

    get_result(k, k_derivative, v, q, f, left_condition, right_condition, true_solution=np.vectorize(u), x_bias=0.01)

if __name__ == "__main__":
    main()