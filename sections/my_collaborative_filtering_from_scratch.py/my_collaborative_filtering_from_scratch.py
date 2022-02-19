from typing import List
import pandas as pd
import numpy as np
from pathlib2 import Path

DATA_PATH = Path('./data/rating.csv')
# TRAIN_SIZE = int(.8e6)
# TEST_DATA = int(.2e6)
TRAIN_SIZE = int(.8e4)
TEST_SIZE = int(.2e4)

print('Getting train Data')
train_data: pd.DataFrame = pd.read_csv(
    DATA_PATH, delimiter=',', nrows=TRAIN_SIZE, index_col=["userId", "movieId"])
train_data.dropna(axis=0, inplace=True)

print('Getting test data...')
test_data: pd.DataFrame = pd.read_csv(DATA_PATH, delimiter=',', index_col=[
                                      "userId", "movieId"], nrows=TEST_SIZE, skiprows=range(1, TRAIN_SIZE))
test_data.dropna(axis=0, inplace=True)

print('Building model...')


def addDeviationData(df: pd.DataFrame):
    df_cp = df.copy()
    df_cp["dev_rating"] = 0
    userIds = df_cp.index.unique(level=0).values.tolist()
    for userId in userIds:
        userId_df = df_cp[df_cp.index.get_level_values(level=0) == userId]
        user_mean_score = userId_df["rating"].mean()
        df_cp.loc[userId, "dev_rating"] = (df_cp.loc[userId]["rating"] - user_mean_score).values
    return df_cp

def getMoviesByUserId(df: pd.DataFrame, userId):
    return df[df.index.get_level_values(0) == userId].index.get_level_values(1).values


def cosSimilarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# TODO: finalizar
def calcSimilarity(data: pd.DataFrame, userId_1, userId_2):

    # TODO: 1. Descobrir movieIds comuns entre 1 e 2
    user1Movies = getMoviesByUserId(data, userId_1)
    user2Movies = getMoviesByUserId(data, userId_2)
    commomMovies = list(set(user1Movies) & set(user2Movies))

    # print(data.loc[[userId_1, userId_2]])
    # print(data[np.in1d(data.index.get_level_values(1), commomMovies)])
    if len(commomMovies) == 0:
        return 0

    subData = data.loc[[userId_1, userId_2]]
    subData = subData[np.in1d(subData.index.get_level_values(1), commomMovies)]
    dev_ratings1 = subData.loc[userId_1]["dev_rating"].values
    dev_ratings2 = subData.loc[userId_2]["dev_rating"].values
    
    # TODO: 3. calcular cosineSimilarity com os .value de cada
    # return cosSimilarity(..., ...)
    return cosSimilarity(dev_ratings1, dev_ratings2)


class CollaborativeFilteringModel:
    def __init__(self) -> None:
        self.weights = {}

    def fit(self, data: pd.DataFrame):
        userIds = data.index.unique(level="userId").values
        for i, userId_1 in enumerate(userIds[:-1]):
            for userId_2 in userIds[i+1:]:
                self.weights[(userId_1, userId_2)] = calcSimilarity(
                    data, userId_1, userId_2)
        print(self.weights)


    # TODO: implementar
    def predict(self, data, userId, movieId):

        other_userIds: List = df_cp.index.unique(level=0).values.tolist().remove(userId)

        userId_df = data[data.index.get_level_values(level=0) == userId]
        user_mean_score = userId_df["rating"].mean()

        return np.ndarray([self.weights[(userId, i)] * train_data.loc[(1, 29)]["dev_rating"] for i in other_userIds])


train_data = addDeviationData(train_data)
print(train_data.loc[(1, 29)]["dev_rating"])
assert False
model = CollaborativeFilteringModel()
model.fit(train_data)
model.predict()
