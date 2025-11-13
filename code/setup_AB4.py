import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt

import template.output as output
import template.questions as questions
import template.text_order_widget as text_order_widget
import template.table_widget as table


def als_with_mask(matrix, num_factors=2, reg_param=0, max_iter=100):
    """
    Alternating Least Squares (ALS) algorithm for matrix factorization with a mask.

    Parameters:
    - matrix: numpy array
        The target matrix to be factorized.
    - num_factors: int, optional (default=2)
        Number of latent factors.
    - reg_param: float, optional (default=0.01)
        Regularization parameter.
    - max_iter: int, optional (default=100)
        Maximum number of iterations.

    Returns:
    - U: numpy array
        User factors matrix.
    - M: numpy array
        Item factors matrix.
    """

    # Create a mask indicating observed (non-zero) values
    mask = np.where(matrix != 0, 1, 0)

    # Get the shape of the matrix
    num_users, num_items = matrix.shape

    # Initialize user and item factors matrices with random values
    U = np.random.rand(num_users, num_factors)
    M = np.random.rand(num_factors, num_items)

    # ALS algorithm iterations
    error_list = []
    for x in range(max_iter):
        # Update user factors U
        for i in range(num_users):
            observed_indices = np.where(mask[i] == 1)[0]
            M_T_M = np.dot(M[:, observed_indices], M[:, observed_indices].T) + reg_param * np.eye(num_factors)
            M_T_R = np.dot(M[:, observed_indices], matrix[i, observed_indices].T)
            U[i] = np.linalg.solve(M_T_M, M_T_R)

        # Update item factors M
        for j in range(num_items):
            observed_indices = np.where(mask[:, j] == 1)[0]
            U_T_U = np.dot(U[observed_indices].T, U[observed_indices]) + reg_param * np.eye(num_factors)
            U_T_R = np.dot(U[observed_indices].T, matrix[observed_indices, j])
            M[:, j] = np.linalg.solve(U_T_U, U_T_R)

        # Reconstruct the target matrix
        predicted_matrix = np.dot(U, M)
        error_value = calcError(matrix, predicted_matrix)
        error_list.append(error_value)
    return U, M, predicted_matrix, error_list


def gradient_descent_with_mask_(matrix, num_factors=2, reg_param=0, learning_rate=0.01, max_iter=100):
    """
    Gradient Descent algorithm for matrix factorization with a mask.

    Parameters:
    - matrix: numpy array
        The target matrix to be factorized.
    - num_factors: int, optional (default=2)
        Number of latent factors.
    - reg_param: float, optional (default=0.01)
        Regularization parameter.
    - learning_rate: float, optional (default=0.01)
        Learning rate for gradient descent.
    - max_iter: int, optional (default=100)
        Maximum number of iterations.

    Returns:
    - U: numpy array
        User factors matrix.
    - M: numpy array
        Item factors matrix.
    """

    # Create a mask indicating observed (non-zero) values
    mask = np.where(matrix != 0, 1, 0)

    # Get the shape of the matrix
    num_users, num_items = matrix.shape

    # Initialize user and item factors matrices with random values
    U = np.random.rand(num_users, num_factors)
    M = np.random.rand(num_factors, num_items)

    # Gradient Descent algorithm iterations
    error_list = []
    for x in range(max_iter):
        # Update user factors U
        for i in range(num_users):
            observed_indices = np.where(mask[i] == 1)[0]
            predicted_ratings = np.dot(U[i, :], M[:, observed_indices])
            error = matrix[i, observed_indices] - predicted_ratings
            gradient_U = -2 * np.dot(error, M[:, observed_indices].T) + 2 * reg_param * U[i, :]
            U[i, :] -= learning_rate * gradient_U

        # Update item factors M
        for j in range(num_items):
            observed_indices = np.where(mask[:, j] == 1)[0]
            predicted_ratings = np.dot(U[observed_indices, :], M[:, j])
            error = matrix[observed_indices, j] - predicted_ratings
            gradient_M = -2 * np.dot(U[observed_indices, :].T, error) + 2 * reg_param * M[:, j]
            M[:, j] -= learning_rate * gradient_M

        # Reconstruct the target matrix
        predicted_matrix = np.dot(U, M)
        error_value = calcError(matrix, predicted_matrix)
        error_list.append(error_value)

    return U, M, predicted_matrix, error_list


def calcError(matrix, predicted_matrix):
    error = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                error = np.abs(matrix[i][j] - predicted_matrix[i][j])**2 / np.prod(matrix.shape)
    return error


def print_rounded_matrix(matrix):
    for row in matrix:
        for element in row:
            print(round(element, 2), end=" ")
        print()


def print_results(R1, U, M, P, error):
    # Print results
    print("Originale Matrix:")
    print_rounded_matrix(R1)
    print("\nVorhersagen Matrix:")
    print_rounded_matrix(P)
    print("\nMovie Matrix:")
    print_rounded_matrix(M)
    print("\nUser Matrix:")
    print_rounded_matrix(U)
    print("\nFehler auf bekannten Daten:")
    print(error[len(error) - 1])


def plotMSEError(error_list):
    # Erstelle einen Plot
    plt.plot(range(1, len(error_list) + 1), error_list, marker='o', linestyle='-')

    # Beschriftungen für die Achsen und den Plot
    plt.title('Error Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Error Value')

    # Zeige den Plot an
    plt.show()


def checkErrorU(error_u_cal):
    errorU = error_u_cal(1, 1)
    if (errorU is None):
        output.wrong("Setze eine Formel ein")
        return
    u1 = np.array([2, -2, -3, 3, 0, 10])
    u2 = np.array([1, 2, -3, 2, 0.1, -0.4])

    eps = 0.01

    def error_u_sol(u11, u12):
        return (2 - (u11 + u12))**2 + (3 - (u11 + u12))**2

    cal = np.array([error_u_cal(u11, u12) for u11, u12 in zip(u1, u2)])
    sol = np.array([error_u_sol(u11, u12) for u11, u12 in zip(u1, u2)])

    if np.max(np.abs(cal - sol)) > eps:
        output.wrong("Deine Lösung für die Fehlerfunktion ist noch nicht korrekt.\n")
    else:
        output.success("Deine Lösung für die Fehlerfunktion ist korrekt.\n")
        plot_error(error_u_cal, -1, 3)


def check_gradient_u(error_gradient1, error_gradient2):

    u1 = np.array([2, -2, -3, 3, 0, 10])
    u2 = np.array([1, 2, -3, 2, 0.1, -0.4])

    eps = 0.01

    def error_gradient1_sol(u11, u12):
        return -2 * (2 - u11 - u12) - 2 * (3 - u11 - u12)

    def error_gradient2_sol(u11, u12):
        return -2 * (2 - u11 - u12) - 2 * (3 - u11 - u12)

    # u11
    if (error_gradient1 is None):
        output.wrong("Setze eine Formel ein")
        return
    cal1 = np.array([error_gradient1(u11, u12) for u11, u12 in zip(u1, u2)])
    sol1 = np.array([error_gradient1_sol(u11, u12) for u11, u12 in zip(u1, u2)])

    if np.max(np.abs(cal1 - sol1)) > eps:
        output.wrong("Deine Lösung für die Ableitung nach u11 ist nicht noch nicht korrekt.")
    else:
        output.success("Deine Lösung für die Ableitung nach u11 ist korrekt.")

    # u12
    if (error_gradient2 is None):
        output.wrong("Setze eine Formel ein")
        return
    cal2 = np.array([error_gradient2(u11, u12) for u11, u12 in zip(u1, u2)])
    sol2 = np.array([error_gradient2_sol(u11, u12) for u11, u12 in zip(u1, u2)])

    if np.max(np.abs(cal2 - sol2)) > eps:
        output.wrong("Deine Lösung für die Ableitung nach u12 ist nicht noch nicht korrekt.")
    else:
        output.success("Deine Lösung für die Ableitung nach u12 ist korrekt.")


def check_min_u(u11_min, u12_min, error_u, error_gradient1, error_gradient2):

    eps = 0.0001

    if abs(error_gradient1(u11_min, u12_min)) < eps and abs(error_gradient2(u11_min, u12_min)) < eps:
        output.success("Dein Ergebnis ist korrekt.")
        
        # Evaluate the error function at the minimum
        error = error_u(u11_min, u12_min)
        print(f"Einsetzen der gefundenen Lösung u11 = {u11_min} und u12 = {u12_min} in die Fehlerfunktion liefert einen Fehler von {error}.")
    else:
        output.wrong("Deine gefundenen Einträge von U liefern noch nicht das Minimum der Fehlerfunktion.")




def check_min_m(m11_min, m12_min, m21_min, m22_min, error_m, error_gradient_m11, error_gradient_m12, error_gradient_m21, error_gradient_m22):

    eps = 0.0001

    if (abs(error_gradient_m11(m11_min, m12_min, m21_min, m22_min)) < eps and 
        abs(error_gradient_m12(m11_min, m12_min, m21_min, m22_min)) < eps and 
        abs(error_gradient_m21(m11_min, m12_min, m21_min, m22_min)) < eps and 
        abs(error_gradient_m22(m11_min, m12_min, m21_min, m22_min)) < eps):
        
        output.success("Dein Ergebnis ist korrekt.")
        
        # Evaluate the error function at the minimum
        error = error_m(m11_min, m12_min, m21_min, m22_min)
        print(f"Einsetzen der gefundenen Lösung m11 = {m11_min}, m12 = {m12_min}, m21 = {m21_min} und m22 = {m22_min} in die Fehlerfunktion liefert einen Fehler von {error}.")
    else:
        output.wrong("Deine gefundenen Werte für die Einträge von U liefern noch nicht das Minimum der Fehlerfunktion.")

        
def check_gradient_m(error_gradient_m11, error_gradient_m12, error_gradient_m21, error_gradient_m22, u1min, u2min):

    m1 = np.array([2, -2, -3, 3, 0, 10])
    m2 = np.array([1, 2, -3, 2, 0.1, -0.4])
    m3 = np.array([0, 2, -3, 2, 0.1, -0.4])
    m4 = np.array([-10, 2, -3, 2, 2.1, 0.4])
    
    eps = 0.01

    def error_gradient_m11_sol(m11, m12, m21, m22):
        return 2 * (2 - u1min * m11 - u2min * m21) * (-u1min)

    def error_gradient_m12_sol(m11, m12, m21, m22):
        return 2 * (3 - u1min * m12 - u2min * m22) * (-u1min)

    def error_gradient_m21_sol(m11, m12, m21, m22):
        return 2 * (2 - u1min * m11 - u2min * m21) * (-u2min)

    def error_gradient_m22_sol(m11, m12, m21, m22):
        return 2 * (3 - u1min * m12 - u2min * m22) * (-u2min)

    # m11
    if np.isnan(error_gradient_m11(1, 2, 3, 4)):
        output.wrong("Gib die partielle Ableitung nach m11 für das erste NaN ein.")
    else:
        cal1 = np.array([error_gradient_m11(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        sol1 = np.array([error_gradient_m11_sol(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        
        if np.max(np.abs(cal1 - sol1)) > eps:
            output.wrong("Deine Lösung für die Ableitung nach m11 ist nicht noch nicht korrekt.")
        else:
            output.success("Deine Lösung für die Ableitung nach m11 ist korrekt.")

    # m12
    if np.isnan(error_gradient_m12(1, 2, 3, 4)):
        output.wrong("Gib die partielle Ableitung nach m12 für das zweite NaN ein.")
    else:
        cal2 = np.array([error_gradient_m12(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        sol2 = np.array([error_gradient_m12_sol(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        
        if np.max(np.abs(cal2 - sol2)) > eps:
            output.wrong("Deine Lösung für die Ableitung nach m12 ist nicht noch nicht korrekt.")
        else:
            output.success("Deine Lösung für die Ableitung nach m12 ist korrekt.")

    # m21
    if np.isnan(error_gradient_m21(1, 2, 3, 4)):
        output.wrong("Gib die partielle Ableitung nach m21 für das dritte NaN ein.")
    else:
        cal3 = np.array([error_gradient_m21(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        sol3 = np.array([error_gradient_m21_sol(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        
        if np.max(np.abs(cal3 - sol3)) > eps:
            output.wrong("Deine Lösung für die Ableitung nach m21 ist nicht noch nicht korrekt.")
        else:
            output.success("Deine Lösung für die Ableitung nach m21 ist korrekt.")

    # m22
    if np.isnan(error_gradient_m22(1, 2, 3, 4)):
        output.wrong("Gib die partielle Ableitung nach m22 für das vierte NaN ein.")
    else:
        cal4 = np.array([error_gradient_m22(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        sol4 = np.array([error_gradient_m22_sol(m1[i], m2[i], m3[i], m4[i]) for i in range(len(m1))])
        
        if np.max(np.abs(cal4 - sol4)) > eps:
            output.wrong("Deine Lösung für die Ableitung nach m22 ist nicht noch nicht korrekt.")
        else:
            output.success("Deine Lösung für die Ableitung nach m22 ist korrekt.")

def print_UTimesM(u11_min, u12_min, m11_min, m12_min, m21_min, m22_min):
    U = np.array([[u11_min, u12_min]])
    M = np.array([[m11_min, m12_min], [m21_min, m22_min]])
    
    print("\nDeine Zerlegung in U und M lautet:")
    print("\nUsermatrix U:")
    print(U)
    
    print("\nMovie-Matrix M:")
    print(M)
    
    P = np.dot(U, M)
    
    print("\nDas Produkt U*M und damit die Vorhersagematrix P lautet:")
    print(P)

def plot_error(error_u_cal, start, end):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    u11 = np.linspace(start, end, 100)
    u12 = np.linspace(start, end, 100)
    U11, U12 = np.meshgrid(u11, u12)
    Z = np.array([error_u_cal(u11, u12) for u11, u12 in zip(np.ravel(U11), np.ravel(U12))])
    Z = Z.reshape(U11.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U11, U12, Z, cmap='viridis')
    ax.set_xlabel('u11')
    ax.set_ylabel('u12')
    ax.set_zlabel('Error')
    plt.show()


def promt_4_2_a1():
    questions.prompt_answer("WS4-2a1", input_prompt="Wert für $\lambda$", input_description="Deine Antwort")


def promt_4_2_a2():
    questions.prompt_answer("WS4-2a2", input_prompt="Werte für U und M", input_description="Deine Antwort")
