import numpy as np
import pandas as pd


def _ornstein_uhlenbeck_anomaly(
    t,
    size=1,
    noise_scale=1,
    mean_reverting=1,
    anomaly_freq=1.0,
    anomaly_duration=0.1,
    anomaly_scale=1,
    seed=314,
):
    rng = np.random.RandomState(seed)
    (length,) = t.shape
    y = np.zeros(length)
    X = np.empty((length, size))
    X[0] = 0
    target = np.zeros(size)
    for i in range(1, length):
        dt = t[i] - t[i - 1]
        target *= np.exp(-dt / anomaly_duration)
        X[i] = rng.normal(
            target + (X[i - 1] - target) * np.exp(-mean_reverting * dt),
            noise_scale * dt**0.5,
        )
        if rng.rand() < dt / anomaly_freq:
            y[i] = 1
            target += rng.normal(scale=anomaly_scale, size=size)
    return X, y


def synthetise_ornstein_uhlenbeck_data(n_rows=1000, n_cols_x=5):
    t = np.arange(n_rows)
    t = t / t.shape[0]
    X, y = _ornstein_uhlenbeck_anomaly(
        t,
        size=n_cols_x,
        noise_scale=1,
        mean_reverting=3,
        anomaly_freq=0.1,
        anomaly_duration=0.005,
        anomaly_scale=40,
    )
    t = pd.to_datetime("2021-01-01") + pd.Timedelta(1, "h") * np.arange(t.shape[0])
    X = pd.DataFrame(X, index=t, columns=[f"X{i}" for i in range(X.shape[1])])
    y = pd.DataFrame(y[:, None], index=t, columns=["y"])
    return X, y
