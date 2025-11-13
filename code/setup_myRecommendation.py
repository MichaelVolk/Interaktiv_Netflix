import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_netflix_data, subset_top_movies, subset_random_users, split_data, als_with_mask, compute_recommendation

col = np.empty(15, dtype=int)
rating = np.zeros((15, 1))

def load_and_split_data():
    movieIndex, userIndex, ratings = load_netflix_data()
    movie_subset, user_subset, ratings_subset = subset_top_movies(movieIndex, userIndex, ratings, 120000)
    movie_subset, user_subset, ratings_subset = subset_random_users(movie_subset, user_subset, ratings_subset, 6500)
    Rtrain, Rtest, ids = split_data(movie_subset, user_subset, ratings_subset, 0.1)
    print_dataset_info_train_test(Rtrain, Rtest)
    return Rtrain, Rtest, ids

def print_dataset_info_train_test(Rtrain, Rtest):
    print(f"The dataset contains\n{Rtrain.shape[1]} movies\n{Rtrain.shape[0]} users\n{np.count_nonzero(Rtrain)} ratings in the training set\n{np.count_nonzero(Rtest)} ratings in the test set\n")

def create_user_vec(Rtrain, col, rating):
    user_vec = np.zeros(Rtrain.shape[1])
    for i in range(len(col)):
        user_vec[col[i]] = rating[i]
    return user_vec

def create_extended_dataset(Rtrain, Rtest, col, rating):
    user_vec = create_user_vec(Rtrain, col, rating)
    Rtrain = np.vstack([Rtrain, user_vec])
    Rtest = np.vstack([Rtest, np.zeros(Rtest.shape[1])])
    return Rtrain, Rtest
