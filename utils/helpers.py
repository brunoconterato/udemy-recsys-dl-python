from tqdm import tqdm
import numpy as np
import pandas as pd


def getUserIds(df: pd.DataFrame) -> np.ndarray:
    userIds = df.index.unique(level=0).values
    userIds.sort()
    return userIds


def getMovieIds(df: pd.DataFrame) -> np.ndarray:
    movieIds = df.index.unique(level=1).values
    movieIds.sort()
    return movieIds


def getUserIndex(df: pd.DataFrame, userId) -> np.ndarray:
    i, = np.where(getUserIds(df) == userId)
    return i[0]


def getMovieIndex(df: pd.DataFrame, movieId) -> np.ndarray:
    i, = np.where(getMovieIds(df) == movieId)
    return i[0]


def addDeviationData(df: pd.DataFrame):
    df_cp = df.copy()
    df_cp["dev_rating"] = 0
    userIds = df_cp.index.unique(level=0).values.tolist()
    for userId in tqdm(userIds):
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


def getUsersThatRatedMovie(df: pd.DataFrame, movieId):
    return df[df.index.get_level_values(1) == movieId].index.get_level_values(0).values.tolist()
