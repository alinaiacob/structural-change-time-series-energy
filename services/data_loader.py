import pandas as pd
from utils import addDatesCol, checkNanValues, cleanData

my_dict = {
    "NXT":"../datasets/nxt_us_d.csv",
    "NEE":"../datasets/nee_us_d.csv",
    "LEU":"../datasets/leu_us_d.csv",
    "UUUU":"../datasets/uuuu_us_d.csv",
    "ENPH":"../datasets/enph_us_d.csv",
    "SPX":"../datasets/^spx_d.csv"
}


def loadData(symbol):
    path = my_dict[symbol]
    df = pd.read_csv(path)
    df = addDatesCol(df)
    df = checkNanValues(df)
    df = cleanData(df)
    return df
