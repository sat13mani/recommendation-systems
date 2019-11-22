import numpy
import pickle

import math

from .preprocessor import preprocess
from .config import train_bin_path


class LatentFactor:
    """
    Latent Factor Model describing all the various
    factors required.
    """

    def __init__(self, alpha=0.1, beta=0.01, k=2, epoch=20):
        """
        :param
        alpha: learning rate for stochastic gradient descent
        beta: regularisation constant for penalising magnitudes
        k: number of hidden factors used while factorising
        epoch: number of iterations performed for stochastic gradient descent
        """
        self.learning_rate = alpha
        self.regularisation_const = beta
        self.utility_matrix = preprocess()
        self.num_factors = k
        self.num_epochs = epoch

        self.num_users = len(self.utility_matrix)
        self.num_items = len(self.utility_matrix[0])

        self.all_ratings = numpy.array(self.get_train_tuple())
        self.global_avg_rating = numpy.mean(self.all_ratings[:, 2])

    def get_train_tuple(self):
        """Loads training dataset from the binary file"""
        with open(train_bin_path, 'rb') as f:
            return pickle.load(f)

    def fit(self):
        """Trains our model using the training data."""
        # dimension : u X k
        user_matrix = numpy.random.normal(
            scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        item_matrix = numpy.random.normal(
            scale=1./self.num_factors, size=(self.num_items, self.num_factors))

        user_bias = numpy.zeros(self.num_users)
        item_bias = numpy.zeros(self.num_items)

        for epoch in range(self.num_epochs):
            temp_util_matrix = numpy.matmul(
                user_matrix, numpy.transpose(item_matrix))

            for user, movie, rating in self.all_ratings:

                error = (rating - temp_util_matrix[user - 1][movie - 1] -
                         self.global_avg_rating - user_bias[user - 1] - item_bias[movie - 1])

                temp_user_matrix = user_matrix[user - 1, :]

                user_matrix[user - 1, :] += self.learning_rate * (
                    error * item_matrix[movie - 1, :] - self.regularisation_const * user_matrix[user - 1, :])
                item_matrix[movie - 1, :] += self.learning_rate * (
                    error * temp_user_matrix - self.regularisation_const * item_matrix[movie - 1, :])

                user_bias[user - 1] += (self.learning_rate *
                                        (error - self.regularisation_const * user_bias[user - 1]))
                item_bias[movie - 1] += (self.learning_rate *
                                         (error - self.regularisation_const * item_bias[movie - 1]))

        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        self.user_bias = user_bias
        self.item_bias = item_bias

    def predict(self, i, j):
        """Returns the predicted value by the model"""
        return (
            self.user_bias[i] +
            self.item_bias[j] +
            self.global_avg_rating +
            self.user_matrix[i, :].dot(self.item_matrix[j, :].T)
        )

    def get_rms_error(self):
        """Returns the Root Mean Square Error of the model"""
        error = 0
        predicted_matrix = self.get_utility_matrix()
        N = len(self.all_ratings)

        for rating_tuple in self.all_ratings:
            user, movie, rating = rating_tuple

            residual = rating - predicted_matrix[user - 1, movie - 1]
            error += pow(residual, 2)

        return math.sqrt(error/N)

    def get_mean_abs_error(self):
        """Returns the Mean Absolute Error of the model"""
        error = 0
        predicted_matrix = self.get_utility_matrix()
        N = len(self.all_ratings)

        for rating_tuple in self.all_ratings:
            user, movie, rating = rating_tuple

            residual = rating - predicted_matrix[user - 1, movie - 1]
            error += math.fabs(residual)

        return error/N

    def get_utility_matrix(self):
        """Returns the predicted utility matrix after training the model"""
        return (
            self.global_avg_rating +
            self.user_bias[:, numpy.newaxis] +
            self.item_bias[numpy.newaxis:, ] +
            self.user_matrix.dot(self.item_matrix.T))

    def __str__(self):
        """Returns the string representation of the model"""
        return str(self.get_utility_matrix())
