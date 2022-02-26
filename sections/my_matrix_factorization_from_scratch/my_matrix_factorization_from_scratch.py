from getpass import getuser
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.dataUtils import loadData
from utils.helpers import addDeviationData, getMovieIds, getMovieIndex, getMoviesByUserId, getUserIds, getUserIndex, getUsersThatRatedMovie

K = 5
NROWS = int(1e4)
T = 20

data = loadData(nrows=NROWS)
data = addDeviationData(data)

N = len(getUserIds(data))
M = len(getMovieIds(data))


W = 2 * np.random.random((N, K)) - 1
U = 2 * np.random.random((M, K)) - 1


def solveForWi(data, userId):
    movieIds = getMoviesByUserId(data, userId)

    A = np.zeros((K, K), dtype=np.float64)
    B = np.zeros((K, ), dtype=np.float64)
    for movieId in movieIds:
        movieIndex = getMovieIndex(data, movieId=movieId)
        u_j = U[movieIndex, :]

        r_ij = data.loc[(userId, movieId)]["dev_rating"]
        A += np.outer(u_j, u_j)
        B += r_ij * u_j
    return np.linalg.solve(A, B)


def solveForUj(data, movieId):
    userIds = getUsersThatRatedMovie(data, movieId=movieId)

    A = np.zeros((K, K), dtype=np.float64)
    B = np.zeros((K, ), dtype=np.float64)
    for userId in userIds:
        userIndex = getUserIndex(data, userId=userId)
        w_i = W[userIndex, :]

        r_ij = data.loc[(userId, movieId)]["dev_rating"]
        A += np.outer(w_i, w_i)
        B += r_ij * w_i
    return np.linalg.solve(A, B)


def getTotalLoss(data):
    l = 0.0
    for userId in getUserIds(data):
        i = getUserIndex(data, userId=userId)
        for movieId in getMoviesByUserId(data, userId):
            j = getMovieIndex(data, movieId=movieId)
            l += (data.loc[userId, movieId]
                  ["dev_rating"] - np.dot(W[i], U[j])) ** 2
    return l


def getLoss(data, userId, movieId):
    i = getUserIndex(data, userId=userId)
    j = getMovieIndex(data, movieId=movieId)
    l = (data.loc[userId, movieId]["dev_rating"] - np.inner(W[i, :], U[j, :])) ** 2
    return l


losses = []


def fitWandU(data):
    total_epoch_ops = len(getUserIds(data)) + len(getMovieIds(data))
    for t in range(T):
        print(f'\n\n Starting iteration {t + 1} out of {T}')
        with tqdm(total=total_epoch_ops) as pbar:
            for userId in getUserIds(data):
                i = getUserIndex(data, userId=userId)
                W[i, :] = solveForWi(data, userId=userId)
                pbar.update(1)

            for movieId in getMovieIds(data):
                j = getMovieIndex(data, movieId=movieId)
                U[j, :] = solveForUj(data, movieId=movieId)
                pbar.update(1)

            print('W[66, :]', W[66, :])
            print('U[66, :]', U[66, :])


def getReccomendation4(data, userId, movieId):
    return np.dot(
        W[getUserIndex(data, userId=userId), :],
        U[getMovieIndex(data, movieId=movieId), :]
    )


def plotLoss(losses):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(losses)
    plt.show()


def run():
    userIds = getUserIds(data)
    fitWandU(data)
    getReccomendation4(data, 1, 29)
    # plotLoss(losses)
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
