from getpass import getuser
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.dataUtils import loadData
from utils.helpers import addDeviationData, getMovieIds, getMovieIndex, getMoviesByUserId, getUserIds, getUserIndex, getUsersThatRatedMovie

K = 10
NROWS = int(1e5)
T = 20

data = loadData(nrows=NROWS)
data = addDeviationData(data)

N = len(getUserIds(data))
M = len(getMovieIds(data))


W = 5 * np.random.random((N, K))
U = 5 * np.random.random((M, K))


def solveForWi(data, userId):
    userIndex = getUserIndex(data, userId=userId)
    movieIds = getMoviesByUserId(data, userId)

    A = np.zeros((K, K))
    B = np.zeros((K, ))
    for movieId in movieIds:
        movieIndex = getMovieIndex(data, movieId=movieId)
        u_j = U[movieIndex, :]

        r_ij = data.loc[(userId, movieId)]["dev_rating"]
        A = np.add(A, np.outer(u_j, u_j))
        B += r_ij * u_j
    W[userIndex, :] = np.linalg.solve(A, B)


def solveForUj(data, movieId):
    movieIndex = getMovieIndex(data, movieId=movieId)
    userIds = getUsersThatRatedMovie(data, movieId=movieId)

    A = np.zeros((K, K))
    B = np.zeros((K, ))
    for userId in userIds:
        userIndex = getUserIndex(data, userId=userId)
        w_i = W[userIndex, :]

        r_ij = data.loc[(userId, movieId)]["dev_rating"]
        A = np.add(A, np.outer(w_i, w_i))
        B += r_ij * w_i
    U[movieIndex, :] = np.linalg.solve(A, B)


def getTotalLoss(data):
    l = 0.0
    for userId in getUserIds(data):
        i = getUserIndex(data, userId=userId)
        for movieId in getMoviesByUserId(data, userId):
            j = getMovieIndex(data, movieId=movieId)
            l += (data.loc[userId, movieId]["dev_rating"] - np.dot(W[i], U[j])) ** 2
    return l

def getLoss(data, userId, movieId):
    i = getUserIndex(data, userId=userId)
    j = getMovieIndex(data, movieId=movieId)
    l = (data.loc[userId, movieId]["dev_rating"] - np.dot(W[i], U[j])) ** 2
    return l


losses = []

def getReccomendation4(data, userId, movieId):
    # print('W[i, :]',
    #         W[getUserIndex(data, userId), :])
    # print('U[:, j]',
    #       U[:, getMovieIndex(data, movieId)])
    for t in tqdm(range(T)):
        solveForWi(data, userId=userId)
        solveForUj(data, movieId=movieId)
        # print('W[i, :]',
        #       W[getUserIndex(data, userId), :])
        # print('U[:, j]',
        #       U[:, getMovieIndex(data, movieId)])
        if (t > 0):
            losses.append(getLoss(data, userId, movieId))

    w_i = W[getUserIndex(data, userId=userId), :]
    u_j = U[getMovieIndex(data, movieId=movieId), :]
    return np.dot(w_i.T, u_j.T)


def plotLoss(losses):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(losses)
    plt.show()

def run():
    userIds = getUserIds(data)
    getReccomendation4(data, 1, 29)
    plotLoss(losses)
    # for i in range(2, 100):
    #     print(f'\n\nIteration {i} out of 100...')
    #     userId = np.random.choice(userIds)
    #     moviesWatched = getMoviesByUserId(data, userId)
    #     movieId = np.random.choice(moviesWatched)
    #     print(f"Predicting rating for user {userId} and movie {movieId}...")
    #     if (userId, movieId) in data.index:
    #         print(
    #             f'Rating (deviation) user gave to movie: {data.loc[userId, movieId]["dev_rating"]}')
    #     else:
    #         print(f'User {userId} did not rating movie {movieId} yet')
    #     predictedRating = getReccomendation4(data, userId, movieId)
    #     print(
    #         f'Predicted rating (deviation) for user {userId} and movie {movieId}: {predictedRating}\n')

