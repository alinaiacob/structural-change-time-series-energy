import pandas as pd

from services.features import addFeatures
from utils import cumsum


def computeCumsum(symbol):
    df = addFeatures(symbol)
    df = cumsum(df, symbol)
    return df