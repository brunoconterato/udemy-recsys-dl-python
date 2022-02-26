from numpy import number
import pandas as pd
from pathlib2 import Path

DATA_PATH = Path('./data/rating.csv')

def getTopByKey(data: pd.DataFrame, topN: number, key: str):
    return data.reset_index().set_index(keys=key).index.value_counts().index.values[:topN]


def loadData(mode, topUsers=0, topMovies=0, nrows=0, skip=0) -> pd.DataFrame:
    assert mode in ["all", "top", "byFileRows"]
    if mode == "byFileRows":
        assert nrows > 1
        allData: pd.DataFrame = pd.read_csv(DATA_PATH, delimiter=',', index_col=[
                                            "userId", "movieId"], nrows=nrows, skip=skip)
        allData.dropna(axis=0, inplace=True)

    allData: pd.DataFrame = pd.read_csv(
        DATA_PATH, delimiter=',', index_col=["userId", "movieId"])
    allData.dropna(axis=0, inplace=True)
    if mode == "top":
        assert topUsers > 0 and topMovies > 0
        _topUsers = getTopByKey(allData, topUsers, key="userId")
        _topMovies = getTopByKey(allData, topMovies, "movieId")
        return allData.loc[allData.index.isin(_topUsers, level=0) & allData.index.isin(_topMovies, level=1)]

    if mode == "all":
        return allData
