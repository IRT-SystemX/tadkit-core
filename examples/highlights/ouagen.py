import numpy as np


def ornstein_uhlenbeck_anomaly(
    t,
    size=1,
    noise_scale=1,
    mean_reverting=1,
    anomaly_freq=1,
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
