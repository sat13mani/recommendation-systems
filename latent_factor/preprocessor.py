import os
import pickle
import numpy
import pandas
from copy import deepcopy
from sklearn.model_selection import train_test_split

from .config import movie_dataset, rating_dataset, users_dataset, \
    utility_matrix_bin_path, test_bin_path, train_bin_path


def load(filename, column):
    '''
    Reads a binary file to load dataset.
    :param
        filename 
        column Used to label the dataframe
    :return
        Pandas.Dataframe Contains the dataset.
    '''
    with open(filename, 'r', encoding='ISO-8859-1') as f:
        text = str(f.read()).strip().split('\n')
        return pandas.DataFrame.from_records(
            [sentence.split('::') for sentence in text], columns=column)


def get_dataset():
    '''
    Splits given dataset into 70% training and 30% test dataset.
    '''
    rating = load(rating_dataset, column=['uid', 'mid', 'rating', 'time'])
    rating.drop(labels=['time'], axis=1, inplace=True)

    return train_test_split(rating, test_size=0.3)


def generate_utility_matrix():
    '''
    Loads dataset and generates a utility matrix (user X item ratings).
    :returns
        utility_matrix numpy.ndarray
    '''
    movie = load(movie_dataset, column=['mid', 'title', 'genre'])
    user = load(users_dataset, column=[
                'uid', 'sex', 'age', 'occupation', 'zip-code'])

    user.drop(labels=['sex', 'age', 'occupation',
                      'zip-code'], axis=1, inplace=True)

    rate_test, rate_train = get_dataset()

    num_users = list(user['uid'].unique())
    num_movies = list(movie['mid'].unique())
    num_users.sort()
    num_movies.sort()

    utility_matrix = numpy.full((len(num_users), len(num_movies)), 0)
    train_tuple = []

    for index in rate_train.index:
        user_row = num_users.index(rate_train['uid'][index])
        movie_col = num_movies.index(rate_train['mid'][index])
        rating = int(rate_train['rating'][index])
        utility_matrix[user_row][movie_col] = rating

        train_tuple.append((user_row, movie_col, rating))


    utility_matrix = pandas.DataFrame.from_records(
        utility_matrix, index=num_users, columns=num_movies)
    
    save(train_tuple, train_bin_path)
    save(rate_test, test_bin_path)

    return utility_matrix


def save(matrix, file_path):
    '''
    Saves the a matrix in a binary file.
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(matrix, f)


def preprocess():
    '''
    Wrapper function for preprocessing data.
    :return
        utility_matrix numpy.ndarray
    '''
    if not os.path.exists(utility_matrix_bin_path):
        utility_matrix = generate_utility_matrix()
        save(utility_matrix, utility_matrix_bin_path)
    else:
        with open(utility_matrix_bin_path, 'rb') as f:
            utility_matrix = pickle.load(f)
    return utility_matrix


if __name__ == "__main__":
    utility_matrix = preprocess()
