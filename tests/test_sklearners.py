import numpy as np
import pandas as pd
from tadkit.catalog.sklearners import (
    KDEOutlierDetector,
    GMMOutlierDetector,
    CustomScoreOutlierDetector,
)


n_timestamps = 100
n_sensors = 5
timestamps = pd.to_datetime("2024-01-01", utc=True) + pd.Timedelta(1, "h") * np.arange(
    n_timestamps
)
X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)


detector = KDEOutlierDetector(contamination=0.1)
detector.fit(X)
print(detector.predict(X))

detector = GMMOutlierDetector(contamination=0.1)
detector.fit(X)
print(detector.predict(X))


detector_custom = CustomScoreOutlierDetector(
    score_func=lambda X: -np.linalg.norm(X, axis=1)
)
detector_custom.fit(X)
print(detector_custom.predict(X))
