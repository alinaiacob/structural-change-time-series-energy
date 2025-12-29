import pandas as pd
from arch import arch_model
import numpy as np

from services.features import addFeatures


def computeGarch(symbol, window = 500):
    df = addFeatures(symbol)
    returns = df["log_returns"] * 100
    garch_vol = []

    for i in range(window, len(returns)):
        r_window = returns.iloc[i-window:i]
        model = arch_model(
            r_window,
            vol = "GARCH",
            p = 1,
            q = 1,
            mean = "Zero",
            dist = "normal"
        )
        res = model.fit(disp="off")
        sigma_t = np.sqrt(res.forecast(horizon=1)).variance.values[-1, 0]
        garch_vol.append(sigma_t)

        garch_series = pd.Series(
            garch_vol,
            index = returns.index[window:]
        )

        df["garch_vol"] = garch_series
        return df.dropna()


