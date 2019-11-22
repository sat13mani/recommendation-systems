import numpy
import pickle
from .preprocessor import preprocess
from .config import train_bin_path


class LatentFactor:
    '''
    Latent Factor Model describing all the various 
    factors required. 
    '''

    def __init__(self, alpha=0.1, beta=0.01, k=2, epoch=20):
        '''
        Intialises the object.
        '''
        self.learning_rate = alpha
        self.regularisation_const = beta
        self.utility_matrix = preprocess()
        self.num_factors = 3
        self.num_epochs = epoch

        self.num_users = len(self.utility_matrix)
        self.num_items = len(self.utility_matrix[0])

        self.all_ratings = self.get_train_tuple()

    def get_train_tuple(self):
        with open(train_bin_path, 'rb') as f:
            return pickle.load(f)

    def fit(self):
        # dimension : u X k
        user_matrix = numpy.random.normal(
            scale=1./self.num_factors, size=(self.num_users, self.num_factors))
        item_matrix = numpy.random.normal(
            scale=1./self.num_factors, size=(self.num_items, self.num_factors))

        for epoch in range(self.num_epochs):
            temp_util_matrix = numpy.matmul(
                user_matrix, numpy.transpose(item_matrix))
                
            for rating_tuple in self.all_ratings:
                user, movie, rating = rating_tuple

                error = rating - temp_util_matrix[user - 1][movie - 1]

                user_matrix[user - 1, :] += self.learning_rate * (
                    error * item_matrix[movie - 1, :] - self.regularisation_const * user_matrix[user - 1, :])
                item_matrix[movie - 1, :] += self.learning_rate * (
                    error * user_matrix[user - 1, :] - self.regularisation_const * item_matrix[movie - 1, :])

        self.user_matrix = user_matrix
        self.item_matrix = item_matrix

    def __str__(self):
        return str(numpy.matmul(self.user_matrix, numpy.transpose(self.item_matrix)))
