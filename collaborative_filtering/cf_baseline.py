import numpy as np
from time import time
from collections import Counter
import pickle
import os


def loadFile(filename):
    '''
    Loads file saved after running preprocess.py.
    return: opened file object
    '''
    file = open(filename, 'rb')
    filename = pickle.load(file)
    return filename


def meanRating(matrix):
    '''
    calculated mean rating of give parameter matrix
    return: mean_rating calculated
    '''
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
    '''
    Fills utility matrix using baseline approach of collaborative filtering
    return: prediction and rating
    '''
    rating = []
    prediction = []

    for i in range(int(len(test["movie_id"]) / 10)):
        if i % 100 == 0:
            print(i, " ", int(len(test["movie_id"]) / 10))
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
                den += abs(movie_sim[j])
        predicted_rating = b
        if den > 0:
            predicted_rating += num / den

        if (predicted_rating > 5):
            predicted_rating = 5
        elif (predicted_rating < 0):
            predicted_rating = 0
        predicted_rating = predicted_rating
        prediction.append(predicted_rating)
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


def mapGenre():
    '''
    Builds a dictionary with key as movie id and genre as values
    return: built dictionary
    '''
    cwd = os.path.abspath('../')
    cwd = os.path.join(cwd, "dataset")
    filename = os.path.join(cwd, "movies.dat")
    List = []
    with open(filename, 'r') as f:
        data = f.read()
        data = data.split("\n")
        for li in data:
            list_temp = li.split("::")
            if (len(list_temp) > 1):
                List.append(list_temp[2])
    return List


def main():
    l_start = time()
    utility_matrix = loadFile("utility")
    ratings = loadFile("utility")
    test = loadFile("test")
    umap = loadFile("users_map")
    mmap = loadFile("movie_map")
    sim = loadFile("similarity")
    movie = loadFile("movie")
    l_end = time()
    l_time = l_end - l_start

    inv_map = {}
    for k, v in mmap.items():
        inv_map[v] = k

    genre_list = mapGenre()
    user_id = "510"

    comp_start = time()
    umat = np.transpose(utility_matrix)
    mur = meanRating(utility_matrix)
    mmr = meanRating(umat)
    mew = sum(sum(utility_matrix)) / np.count_nonzero(utility_matrix)
    prediction, actual = baseLineFilter(
        umat, sim, mmap, umap, ratings, mur, mmr, test, mew)
    comp_end = time()
    comp_time = comp_end - comp_start

    rmse, mae = computeError(actual, prediction)

    print(f"load time {l_time}")
    print(f"computation time {comp_time}")
    print("root mean square error :: ", rmse)
    print("print mean absolute error :: ",mae)

    recommendations = topKRecommendation(20, mmap, sim, user_id)
    
    print(f"\n\n*******Recommendation for user {user_id}*******\n")
    for item in recommendations:
        # item[1] = movie[inv_map[item[1]]]
        id = inv_map[item[1]]
        print(movie[id], " ", genre_list[int(id)])


if __name__ == "__main__":
    main()
