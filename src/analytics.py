import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf


def descriptive_statistics(series):

    x = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    return pd.Series({
        "Observaciones": len(x),
        "Media": x.mean(),
        "Mediana": x.median(),
        "Desviacion estandar": x.std(),
        "Varianza": x.var(),
        "Minimo": x.min(),
        "Percentil 25": x.quantile(0.25),
        "Percentil 75": x.quantile(0.75),
        "Maximo": x.max(),
        "Asimetria": x.skew(),
        "Curtosis": x.kurt()
    })


def temporal_reviews(
    reviews,
    frequency="MS",
    unique=True
):

    aggregation = "nunique" if unique else "size"

    series = (
        reviews.groupby(
            pd.Grouper(key="date", freq=frequency)
        )["id"]
        .agg(aggregation)
        .sort_index()
    )

    return series


def moving_average(series, window=3):

    return series.rolling(
        window=window,
        min_periods=window
    ).mean()


def temporal_decomposition(series, period=12):

    series = series.dropna()

    if len(series) < 2 * period:
        return None

    if series.nunique() <= 1:
        return None

    result = STL(
        series,
        period=period,
        robust=True
    ).fit()

    return pd.DataFrame({
        "observado": result.observed,
        "tendencia": result.trend,
        "estacionalidad": result.seasonal,
        "residuo": result.resid
    })


def autocorrelation_analysis(series, lags=24):

    series = series.dropna()

    if len(series) < 6 or series.nunique() <= 1:
        return None

    max_lags = min(
        lags,
        len(series) // 2 - 1
    )

    return pd.DataFrame({
        "retardo": np.arange(max_lags + 1),
        "ACF": acf(
            series,
            nlags=max_lags,
            fft=True
        ),
        "PACF": pacf(
            series,
            nlags=max_lags,
            method="ywm"
        )
    })


def available_stay(calendar, checkin, nights):

    start = pd.Timestamp(checkin).normalize()

    end = start + pd.Timedelta(days=nights)

    interval = calendar[
        (calendar["date"] >= start)
        & (calendar["date"] < end)
    ].copy()

    if interval.empty:
        return []

    valid = interval.groupby("listing_id").agg(
        days=("date", "nunique"),
        available=("available_numeric", "sum")
    )

    first_night = interval[
        interval["date"] == start
    ].set_index("listing_id")

    compatible = valid[
        (valid["days"] == nights)
        & (valid["available"] == nights)
    ].index

    restrictions = first_night[
        (first_night["minimum_nights"] <= nights)
        & (first_night["maximum_nights"] >= nights)
    ].index

    return compatible.intersection(restrictions).tolist()
