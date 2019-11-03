import numpy as np
from time import time
from collections import Counter
import pickle


def loadFile(filename):
    '''
    Loads file saved after running preprocess.py.
    return: opened file object
    '''
    file = open(filename, 'rb')
    filename = pickle.load(file)
    return filename


def meanRating(matrix):
    mean_rating = matrix.sum(axis=1)
    counts = Counter(matrix.nonzero()[0])
    n_users = matrix.shape[0]
    for i in range(n_users):
        if i in counts.keys():
            mean_rating[i] = mean_rating[i] / counts[i]
        else:
            mean_rating[i] = 0
    return mean_rating


def baseLineFilter(umat, sim, mmap, umap, ratings, mur, mmr, test, mew):
    rating = []
    prediction = []

    for i in range(int(len(test["movie_id"]) / 100)):
        user = test.iloc[i, 0]
        movie = test.iloc[i, 1]
        stars = int(test.iloc[i, 2])
        movie = mmap[str(movie)]
        user = umap[str(user)]
        rating.append(stars)
        movie_sim = sim[movie]
        user_ratings = umat[:, user]

        b = mmr[movie] + mur[user] - mew

        num, den = 0, 0
        for j in range(sim.shape[0]):
            if (user_ratings[j] != 0):
                bi = mur[user] + mmr[j] - mew
                num += movie_sim[j] * (user_ratings[j] - bi)
                den += movie_sim[j]
        predicted_rating = b
        if den > 0:
            predicted_rating += num / den

        if (predicted_rating > 5):
            predicted_rating = 5
        elif (predicted_rating < 0):
            predicted_rating = 0
        predicted_rating = int(round(predicted_rating))
        prediction.append(predicted_rating)
    print(prediction)
    return prediction, rating


def computeError(actual_rating, prediction):
    '''
    Computes root mean square error and mean absolute error
    return: rmse -- root mean square (float)
            mean -- mean absolute error (float)
    '''
    n = len(prediction)
    actual_rating = np.array(actual_rating)
    prediction = np.array(prediction)
    rmse = np.sum(np.square(prediction - actual_rating)) / n
    mae = np.sum(np.abs(prediction - actual_rating)) / n
    return rmse, mae


def topKRecommendation(k, movie_map, similarity, movie_id):
    '''
    Generates top k recommendations similar to a movie
    return: top_similar -- list of tuples(similarity, movie_no)
    '''
    row_no = movie_map[movie_id]
    top_similar = []
    for i in range(len(movie_map)):
        if (i != row_no):
            top_similar.append((similarity[row_no][i], i))
    top_similar.sort(reverse=True)
    return top_similar[:k]


def main():
    utility_matrix = loadFile("utility")
    ratings = loadFile("utility")
    test = loadFile("test")
    umap = loadFile("users_map")
    mmap = loadFile("movie_map")
    sim = loadFile("similarity")

    umat = np.transpose(utility_matrix)
    mur = meanRating(utility_matrix)
    mmr = meanRating(umat)
    mew = sum(sum(utility_matrix)) / np.count_nonzero(utility_matrix)
    prediction, actual = baseLineFilter(
        umat, sim, mmap, umap, ratings, mur, mmr, test, mew)

    rmse, mae = computeError(actual, prediction)
    print(rmse)
    print(mae)
    recommendations = topKRecommendation(4, mmap, sim, "102")
    print("recommendations for the user ", recommendations)


if __name__ == "__main__":
    main()
