"""
`TADkit`: **Time-series Anomaly Detection kit** is a set of tools for anomaly detection of time series data.

The `tadkit` python package provides interfaces for anomaly detection that allows coherent and concurrent use of the
various time-series anomaly detection approaches developed in Confiance.ai (TDAAD, SBAD, KCPD, CNNDRAD, ...).

"""

from .base import Formalizer, TADLearner
