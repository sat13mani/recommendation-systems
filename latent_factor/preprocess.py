import pickle
import numpy
import pandas
from sklearn.model_selection import train_test_split

from config import movie_dataset, rating_dataset, users_dataset, utility_matrix_bin_path


def load(filename, column):
    with open(filename, 'r', encoding='ISO-8859-1') as f:
        text = str(f.read()).strip().split('\n')
        return pandas.DataFrame.from_records(
            [sentence.split('::') for sentence in text], columns=column)


def get_dataset():
    rating = load(rating_dataset, column=['uid', 'mid', 'rating', 'time'])
    rating.drop(labels=['time'], axis=1, inplace=True)

    return train_test_split(rating, test_size=0.3)


def generate_utility_matrix():
    movie = load(movie_dataset, column=['mid', 'title', 'genre'])
    user = load(users_dataset, column=[
                'uid', 'sex', 'age', 'occupation', 'zip-code'])

    user.drop(labels=['sex', 'age', 'occupation',
                      'zip-code'], axis=1, inplace=True)

    rate_test, rate_train = get_dataset()

    num_users = list(user['uid'].unique())
    num_movies = list(movie['mid'].unique())
    utility_matrix = numpy.full((len(num_users), len(num_movies)), 0)

    for index in rate_train.index:
        user_row = num_users.index(rate_train['uid'][index])
        movie_col = num_movies.index(rate_train['mid'][index])
        rating = rate_train['rating'][index]
        print(user_row, movie_col, rating)
        utility_matrix[user_row][movie_col] = int(rating)

    return utility_matrix


def save(matrix):
    with open(utility_matrix_bin_path, 'wb') as f:
        pickle.dump(matrix, f)


def preprocess():
    utility_matrix = generate_utility_matrix()
    save(utility_matrix)
    return utility_matrix


if __name__ == "__main__":
    utility_matrix = preprocess()
