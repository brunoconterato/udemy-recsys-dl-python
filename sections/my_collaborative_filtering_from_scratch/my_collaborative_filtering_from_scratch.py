import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.dataUtils import loadData
from utils.helpers import addDeviationData, cosSimilarity, getCommomMovies, getMovieIds, getMoviesByUserId, getUserIds, getUsersThatRatedMovie

MIN_COMMOM_MOVIES_THRESOULD = 20

TRAIN_SIZE = int(1e5)


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


class CollaborativeFilteringModel:
    def __init__(self) -> None:
        self.weights = {}
        self.nonZeroWeightsCoount = 0
        self. totalWeights = 0

    @staticmethod
    def _getWeightsIndex(userId1, userId2):
        return (userId1, userId2) if userId1 < userId2 else (userId2, userId1)

    def fitAll(self, data: pd.DataFrame, total_ops):
        self.weights = {}
        self.nonZeroWeightsCoount = 0
        self. totalWeights = 0

        userIds = data.index.unique(level="userId").values
        with tqdm(total=total_ops) as pbar:
            for i, userId_1 in enumerate(userIds[:-1]):
                for userId_2 in userIds[i+1:]:
                    self.weights[self._getWeightsIndex(userId_1, userId_2)] = calcSimilarity(
                        data, userId_1, userId_2)
                    self. totalWeights += 1
                    if self.weights[self._getWeightsIndex(userId_1, userId_2)] > 0:
                        self.nonZeroWeightsCoount += 1
                    pbar.update(1)

    def predict(self, data: pd.DataFrame, userId, movieId):
        print(f"\nPredicting rating for user {userId} and movie {movieId}...")
        if (userId, movieId) in data.index:
            print(
                f'Rating (deviation) user gave to movie: {data.loc[userId, movieId]["dev_rating"]}')
        else:
            print(f'User {userId} did not rating movie {movieId} yet')

        usersRated = getUsersThatRatedMovie(data, movieId)
        if userId in usersRated:
            usersRated.remove(userId)

        try:
            predictedRating = \
                np.array([self.weights[self._getWeightsIndex(userId, i)] * data.loc[(i, movieId)]["dev_rating"] for i in usersRated]).sum() / \
                np.array([self.weights[self._getWeightsIndex(userId, i)]
                          for i in usersRated]).sum()
        except:
            pass

        print(
            f'Predicted rating (deviation) for user {userId} and movie {movieId}: {predictedRating}\n')
        return predictedRating

def run():
    print('Loading data...')
    train_data: pd.DataFrame = loadData(nrows=TRAIN_SIZE)

    print('Adding deviations to data...')
    train_data = addDeviationData(train_data)

    userIds = getUserIds(train_data)
    movieIds = getMovieIds(train_data)

    print('Fitting model...')
    model = CollaborativeFilteringModel()
    model.fitAll(train_data, total_ops=len(userIds)**2-len(userIds))
    print(f'Number of weights: {model.totalWeights}')
    print(
        f'Non zero weights found: {model.nonZeroWeightsCoount} ({100 * model.nonZeroWeightsCoount / model.totalWeights}%)')


    model.predict(train_data, 1, 29)

    for i in range(100):
        userId = np.random.choice(userIds)
        moviesWatched = getMoviesByUserId(train_data, userId)
        movieId = np.random.choice(moviesWatched)
        model.predict(train_data, userId, movieId)
