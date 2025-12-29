import pandas as pd
import numpy as np
from data_loader import loadData
from utils import logReturns, logVolume, volZScore


def addFeatures(symbol):
    df = loadData(symbol)
    df = logReturns(df)
    df = logVolume(df)
    df = volZScore(df)

    df["abs_log_returns"] = df["log_returns"].abs()

    df["rolling20_std"] = df["log_returns"].rolling(20).std()
    df["rolling60_std"] = df["log_returns"].rolling(60).std()
    return df

