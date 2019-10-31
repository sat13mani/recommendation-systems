import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split


cwd = os.path.abspath('../')
cwd = os.path.join(cwd, "dataset")
filename = os.path.join(cwd, "ratings.dat")
List = []
columns = ['user_id', 'movie_id', 'rating', 'timestamp']

with open(filename, 'r') as f:
    data = f.read()
    data = data.split("\n")
    for li in data:
        list_temp = li.split("::")
        List.append(list_temp)

df = pd.DataFrame(List, columns=columns)
test_data, train_data = train_test_split(df, test_size=0.2)
df.drop('timestamp', axis=1, inplace=True)

print(df.head())

movies = df['movie_id'].unique()
users = df['user_id'].unique()
no_movies = len(movies)
no_users = len(users)

print(no_movies, no_users)

movie_map = {}
users_map = {}

''' creating a index mapping for users and movies '''

for k, v in enumerate(movies):
    movie_map[v] = k

for k, v in enumerate(users):
    users_map[v] = k

''' creating two dimensional utility matrix
    rows: users
    columns: movies
'''

utility_mat = np.zeros((no_users, no_movies))

print(df.shape)
print(utility_mat.shape)

print(len(df))
a = 0

for index, row in df.iterrows():
    a += 1
    if a == len(df) - 1:
        break
    utility_mat[users_map[row['user_id']]
                ][movie_map[row['movie_id']]] = int(row['rating'])

for i in range(len(utility_mat[1])):
    print(utility_mat[1][i])

''' persistent storage for the utility matrix and other data '''

file_handler = open("utility", 'wb+')
pickle.dump(utility_mat, file_handler)

file_handler = open("users_map", 'wb+')
pickle.dump(users_map, file_handler)

file_handler = open("movie_map", 'wb+')
pickle.dump(movie_map, file_handler)

file_handler = open("test", 'wb+')
pickle.dump(test_data, file_handler)

file_handler = open("train", 'wb+')
pickle.dump(train_data, file_handler)
