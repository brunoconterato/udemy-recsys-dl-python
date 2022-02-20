from typing import List
import pandas as pd
import numpy as np
from pathlib2 import Path

MIN_COMMOM_MOVIES_THRESOULD = 10

DATA_PATH = Path('./data/rating.csv')
TRAIN_SIZE = int(1e5)


def addDeviationData(df: pd.DataFrame):
    df_cp = df.copy()
    df_cp["dev_rating"] = 0
    userIds = df_cp.index.unique(level=0).values.tolist()
    for userId in userIds:
        userId_df = df_cp[df_cp.index.get_level_values(level=0) == userId]
        user_mean_score = userId_df["rating"].mean()
        df_cp.loc[userId, "dev_rating"] = (
            df_cp.loc[userId]["rating"] - user_mean_score).values
    return df_cp


def getMoviesByUserId(df: pd.DataFrame, userId):
    return df[df.index.get_level_values(0) == userId].index.get_level_values(1).values


def getCommomMovies(data, userId1, userId2):
    user1Movies = getMoviesByUserId(data, userId1)
    user2Movies = getMoviesByUserId(data, userId2)
    return list(set(user1Movies) & set(user2Movies))


def cosSimilarity(v1: np.ndarray, v2: np.ndarray):
    if np.linalg.norm(v1) * np.linalg.norm(v2) < 0.00001:
        return 1 if np.array_equal(v1, v2) else 0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calcSimilarity(data: pd.DataFrame, userId1, userId2):
    commomMovies = getCommomMovies(data, userId1, userId2)

    if len(commomMovies) < MIN_COMMOM_MOVIES_THRESOULD:
        return 0

    if userId1 == userId2:
        return 1

    subData = data.loc[[userId1, userId2]]
    subData = subData[np.in1d(subData.index.get_level_values(1), commomMovies)]
    dev_ratings1 = subData.loc[userId1]["dev_rating"].values
    dev_ratings2 = subData.loc[userId2]["dev_rating"].values

    return cosSimilarity(dev_ratings1, dev_ratings2)


def getUsersThatRatedMovie(df: pd.DataFrame, movieId):
    return df[df.index.get_level_values(1) == movieId].index.get_level_values(0).values.tolist()


class CollaborativeFilteringModel:
    def __init__(self) -> None:
        self.weights = {}
        self.nonZeroWeightsCoount = 0
        self. totalWeights = 0

    @staticmethod
    def _getWeightsIndex(userId1, userId2):
        return (userId1, userId2) if userId1 > userId2 else (userId2, userId1)

    def fitAll(self, data: pd.DataFrame):
        self.weights = {}
        self.nonZeroWeightsCoount = 0
        self. totalWeights = 0

        userIds = data.index.unique(level="userId").values
        for i, userId_1 in enumerate(userIds[:-1]):
            for userId_2 in userIds[i+1:]:
                self.weights[self._getWeightsIndex(userId_1, userId_2)] = calcSimilarity(
                    data, userId_1, userId_2)
                self. totalWeights += 1
                if self.weights[self._getWeightsIndex(userId_1, userId_2)] > 0:
                    self.nonZeroWeightsCoount += 1

    def predict(self, data: pd.DataFrame, userId, movieId):
        print(f"\nPredicting rating for user {userId} and movie {movieId}...")
        if (userId, movieId) in data.index:
            print(
                f'Rating (deviation) user gave to movie {data.loc[userId, movieId]["dev_rating"]}...')
        else:
            print(f'User {userId} did not rating movie {movieId} yet')

        usersRated = getUsersThatRatedMovie(data, movieId)
        usersRated.remove(userId)

        predictedRating = \
            np.array([self.weights[self._getWeightsIndex(userId, i)] * train_data.loc[(i, movieId)]["dev_rating"] for i in usersRated]).sum() / \
            np.array([self.weights[self._getWeightsIndex(userId, i)]
                     for i in usersRated]).sum()

        print(
            f'Predicted rating (deviation) for user {userId} and movie {movieId}: {predictedRating}\n')
        return predictedRating


print('Loading data...')
train_data: pd.DataFrame = pd.read_csv(
    DATA_PATH, delimiter=',', nrows=TRAIN_SIZE, index_col=["userId", "movieId"])
train_data.dropna(axis=0, inplace=True)

print('Adding deviations to data...')
train_data = addDeviationData(train_data)

print('Fitting model...')
model = CollaborativeFilteringModel()
model.fitAll(train_data)
print(f'Number of weights: {model.totalWeights}')
print(
    f'Non zero weights found: {model.nonZeroWeightsCoount} ({100 * model.nonZeroWeightsCoount / model.totalWeights}%)')

model.predict(train_data, 1, 29)
