import numpy as np
import os
import yaml
from tqdm.notebook import tqdm

# Interne Imports
import template.output as output
import template.questions as questions
import template.text_order_widget as text_order_widget
import template.table_widget as table

import setup_language_checks as messages

def load_netflix_data():
    movie_index = np.load("../data/npys/I.npy")
    user_index = np.load("../data/npys/J.npy")
    ratings = np.load("../data/npys/V.npy")

    return movie_index.astype(int), user_index.astype(int), ratings.astype(int)


def print_dataset_info(movie_subset, user_subset, ratings_subset):

    def count_unique_elements(arr):
        return (np.bincount(arr.ravel()) != 0).sum()

    print(f"DatasetInfo {count_unique_elements(movie_subset)} MovieInfo {count_unique_elements(user_subset)} UserInfo {len(ratings_subset)} RatingsInfo")


def print_test_train_info(Rtrain, Rtest):
    print(
        f"DatasetInfo {Rtrain.shape[1]} MovieInfo {Rtrain.shape[0]} UserInfo {np.count_nonzero(Rtrain)} TrainsetInfo {np.count_nonzero(Rtest)} TestsetInfo")


def subset_top_movies(movie_index, user_index, ratings, n_ratings=150000):
    """Create a matrix with all reviews from movies with more than 'n_ratings' ratings."""

    # Check if the number of ratings is in the list of pre-computed datasets
    # If not, load the overview file
    cache_file_path = f"../data/npys/count_{n_ratings}.npy"

    if os.path.isfile(cache_file_path):
        # println("Loading cached dataset.")
        idx = np.load(cache_file_path)
        idx += 1  # Julia starts counting at 1
    else:
        # We search through the overview file to find all movie IDs
        # which have at least n_ratings.
        # Note that the order of idx in the if and else branch
        # will not match.
        overview = yaml.load(open("../data/overview.yml", "r"), Loader=yaml.FullLoader)
        movie_ids = []
        for id, value in overview.items():
            if value["count"] > n_ratings:
                movie_ids.append(id)

        idx = np.where(np.isin(movie_index, movie_ids))[0]

        # Cache file to save time when rerunning
        np.save(cache_file_path, idx)
        print("Caching dataset.")

    sub_movie_index = movie_index[idx]
    sub_user_index = user_index[idx]
    sub_ratings = ratings[idx]

    return sub_movie_index, sub_user_index, sub_ratings


def subset_random_users(movies, users, ratings, n_users):
    np.random.seed(123)
    selected_user_vec = np.random.choice(np.unique(users), n_users, replace=False)
    idx = np.where(np.isin(users, selected_user_vec))[0]

    return movies[idx], users[idx], ratings[idx]


def split_data(movies, users, ratings, test_size=0.1):
    """Split a sparse rating matrix into separate test movie, user, and rating vectors.

    Note that this function replaces existing ratings with zeros.
    """
    if not (0 <= test_size <= 1):
        raise ValueError("Test size must be in the range [0, 1].")

    n_ratings = len(ratings)

    # Sample vector in the range 0 to n_rating with length: test_size * n_ratings
    np.random.seed(111)
    test_idx = np.random.choice(np.arange(n_ratings), size=int(np.floor(test_size * n_ratings)), replace=False)
    train_idx = np.setdiff1d(np.arange(n_ratings), test_idx)

    ratings_train_matrix = np.copy(ratings)
    ratings_test_matrix = np.copy(ratings)

    ratings_train_matrix[test_idx] = 0
    ratings_test_matrix[train_idx] = 0

    train_matrix, movie_to_column, user_to_row = build_rating_matrix(movies, users, ratings_train_matrix)
    # To not have any issues with orderings or randomness, we forward the used user/movie
    # to row/column dictionary
    test_matrix, _, _ = build_rating_matrix(movies, users, ratings_test_matrix, movie_to_column, user_to_row)

    train_matrix, test_matrix, idx = filter_users_without_ratings(train_matrix, test_matrix)
    train_matrix, test_matrix, idx = filter_movies_without_ratings(train_matrix, test_matrix)

    return train_matrix, test_matrix


def build_rating_matrix(movie_all_data, user_all_data, ratings_data, movie_to_column=None, user_to_row=None):
    """Create a rating matrix of shape users x movies."""

    # Create empty matrix for ratings
    unique_movies = np.unique(movie_all_data)
    unique_users = np.unique(user_all_data)
    n_movies = len(unique_movies)
    n_users = len(unique_users)
    rating_matrix = np.zeros((n_users, n_movies), dtype=np.int8)

    # We do not want to assume that the movie vector is grouped/sorted
    # and therefore define a mapping between movie/user to column/row
    if movie_to_column is None:
        movie_to_column = dict(zip(unique_movies, np.arange(1, n_movies+1)))
    if user_to_row is None:
        user_to_row = dict(zip(unique_users, np.arange(1, n_users+1)))

    # Loop over movies and user, look up their row and column and place
    # the rating at this position
    for idx, (m, u) in enumerate(zip(movie_all_data, user_all_data)):
        rating_matrix[user_to_row[u]-1, movie_to_column[m]-1] = ratings_data[idx]

    # We also return the mapping for later usage
    return rating_matrix, movie_to_column, user_to_row


def filter_users_without_ratings(train_matrix, test_matrix):
    temp = np.sum(train_matrix > 0, axis=1)
    idx = np.flatnonzero(temp)
    rm = train_matrix.shape[0] - len(idx)

    print(f"{rm} PrintFilterInfo")

    train_matrix = train_matrix[idx, :]
    test_matrix = test_matrix[idx, :]

    return train_matrix, test_matrix, idx


def filter_movies_without_ratings(train_matrix, test_matrix):
    temp = np.sum(train_matrix > 0, axis=0)
    idx = np.flatnonzero(temp)
    rm = train_matrix.shape[1] - len(idx)

    print(f"{rm} Filme ohne Bewertungen im Trainingsdatensatz wurden entfernt.\n")

    train_matrix = train_matrix[:, idx]
    test_matrix = test_matrix[:, idx]

    return train_matrix, test_matrix, idx


def printMatrix(name, matrix, r, c):
    print(name)
    for row in matrix[:c]:
        line = ""
        for value in row[:r]:
            line = line + str(value) + " "
        print(line)


def print_train_test(Rtrain, Rtest, r, c):
    printMatrix("Trainingsmatrix", Rtrain, r, c)
    printMatrix("Testmatrix", Rtest, r, c)


def print_train_error(train_loss):
    print(f"Der Trainigsfehler ist{train_loss[len(train_loss) - 1]}")


def print_test_error(test_loss):
    print(f"Der Testfehler ist {test_loss}")


def print_train_test_pred(Rtrain, Rtest, P, r, c):
    printMatrix("Trainingsmatrix", Rtrain, r, c)
    printMatrix("Testmatrix", Rtest, r, c)
    printMatrix("Vorhersagematrix", np.round(P, decimals = 2), r, c)


def compute_recommendation(P, user):
    """
    Compute recommendations for a given user based on predicted ratings.

    Parameters:
    - P: numpy array
        Matrix of predicted ratings.
    - user: int
        Index of the user for whom recommendations are needed.

    Returns:
    - recommendations: list
        List of recommended item indices sorted by predicted rating.
    """

    # Get the predicted ratings for the specified user
    user_ratings = P[user, :]

    # Get the indices sorted by predicted rating in descending order
    recommendations = np.argsort(user_ratings)[::-1]
    print("Recommendations for User {}: {}".format(user, recommendations))


# Kopiert aus setup_AB4
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
    for x in tqdm(range(max_iter), desc="Iterationen"):
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


def calcError(matrix, predicted_matrix):
    non_zero_elements = matrix != 0
    squared_diff = (matrix - predicted_matrix)**2

    return np.sum(squared_diff[non_zero_elements]) / np.count_nonzero(non_zero_elements)

def parameter_table():
    table_content = [
        [table.Header("features"), table.Header("lambda"), table.Header("Fehler auf den Trainingsdaten"), table.Header("Fehler auf den Testdaten")],
        [table.Content("8"), table.Content("0.00001"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("0.01"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("0.1"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("0.2"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("0.5"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("1"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("5"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("10"), table.TextInput(), table.TextInput()],
        [table.Content("8"), table.Content("15"), table.TextInput(), table.TextInput()],
        [table.Content("4"), table.Content("0.3"), table.TextInput(), table.TextInput()],
        [table.Content("6"), table.Content("0.3"), table.TextInput(), table.TextInput()],
        [table.Content("12"), table.Content("0.3"), table.TextInput(), table.TextInput()],
        [table.Content("20"), table.Content("0.3"), table.TextInput(), table.TextInput()],
        [table.Content("40"), table.Content("0.3"), table.TextInput(), table.TextInput()],
        [table.TextInput(), table.TextInput(), table.TextInput(), table.TextInput()],
        [table.TextInput(), table.TextInput(), table.TextInput(), table.TextInput()],
        [table.TextInput(), table.TextInput(), table.TextInput(), table.TextInput()],
        [table.TextInput(), table.TextInput(), table.TextInput(), table.TextInput()]
    ]
    column_widths = [100, 100, 100, 100]

    table.show_table("WS2-1d", table_content, column_widths)