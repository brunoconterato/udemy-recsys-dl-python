import pandas as pd
from pathlib2 import Path

DATA_PATH = Path('./data/rating.csv')


def loadData(nrows, skiprows=0) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(
        DATA_PATH, delimiter=',', nrows=nrows, index_col=["userId", "movieId"], skiprows=skiprows)
    data.dropna(axis=0, inplace=True)
    return data
