<div align="center">
    <img src="_static/Logo_ConfianceAI.png" width="20%" alt="Confiance.ai Logo" />
    <h1 style="font-size: large; font-weight: bold;">tadkit-core</h1>
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.12-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MPL-2">
    </a>
    <a href="_static/pylint/pylint.txt">
        <img src="_static/pylint/pylint.svg" alt="Pylint Score">
    </a>
    <a href="_static/flake8/index.html">
        <img src="_static/flake8/flake8.svg" alt="Flake8 Report">
    </a>
	<a href="_static/coverage/index.html">
        <img src="_static/coverage/coverage.svg" alt="Coverage report">
    </a>

</div>
<br>
<div align="center">
    <a href="https://github.com/IRT-SystemX/tadkit-core">
        <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub">
    </a>
    <a href="https://irt-systemx.github.io/tadkit-core/">
        <img src="https://img.shields.io/badge/Online%20Documentation-available-0A66C2?logo=readthedocs&logoColor=white" alt="Docs">
    </a>
    <a href="https://pypi.org/project/tadkit-core/">
        <img src="https://img.shields.io/pypi/v/tadkit-core?color=blue&label=PyPI&logo=pypi&logoColor=white" alt="PyPI">
    </a>
</div>

<br>

---
# TADkit – Timeseries Anomaly Detection kit
Website and documentation : https://irt-systemx.github.io/tadkit-core/


## Overview

`tadkit-core` is a **flexible and extensible Python toolkit for detecting anomalies in time-series data**. It empowers data scientists and developers to quickly identify unusual patterns, monitor system behavior, and build predictive models—all with a modular design that makes integration and customization straightforward.

It builds upon [![scikit-learn](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)](https://scikit-learn.org/) **[scikit-learn](https://scikit-learn.org/)** for interfacing anomaly detection algorithms.


## 🔍 Key Features

- **Unified Interfaces for Anomaly Detection**
  Provides a coherent set of interfaces for different time-series anomaly detection methods. The main abstractions are:
  - `Formater`: prepares raw timeseries data into a machine-learning-friendly format.
  - `TADLearner`: enforces `.fit(X)`, `.score_samples(X)`, and `.predict(X)` coherently for unsupervised anomaly detection.

- **Supports Multiple Detection Methods**
  Includes methods from scikit-learn and Confiance.ai components ([TDAAD](https://catalog.trustworthy-ai-association.eu/records/dkvhy-nk328) and [github](https://github.com/IRT-SystemX/tdaad), [SBAD](https://catalog.confiance.ai/records/fkpja-s7546), [KCPD](https://catalog.trustworthy-ai-association.eu/records/x3vpy-r3587) and [github](https://github.com/confianceai/kernel-change-point-detection), [CNNDRAD](https://catalog.confiance.ai/records/af2ab-hw426), ...). All learners can be instantiated with default parameters.

- **Dynamic Component Loading**
  Only installed components are made available in the system; unavailable components are automatically skipped.

- **Extensible and Modular**
  Designed for easy integration of new anomaly detection methods and smooth scaling across different datasets and applications.


## 🛠 Installation
Install from PyPI (recommended):

```bash
pip install tadkit-core
```
Or install from source:
```bash
git clone https://github.com/IRT-SystemX/tadkit-core.git
cd tadkit-core
pip install -r requirements.txt
```
Requirements:
- Python ≥ 3.8+
- See `requirements.txt` for full dependency list



## 🚀 Quickstart
```python
# Prepare your data
from tadkit.catalog.rawtowideformatter import RawToWideFormatter
formatter = RawToWideFormatter(data=my_raw_data, backend="pandas")
X = formatter.format()

# Query the available anomaly detection methods that are compatible with your data (univariate or multivariate, etc.)
from tadkit.base.registry import registry
for learner_cls in registry.match_learners(formatter):
    learner = learner_cls()  # instantiate directly
    # Learner calibration
    learner.fit(X)
    # Anomaly scores
    y_score = learner.score_samples(X)
    # Detect anomalies
    predictions = learner.predict(X)
```

The modular architecture allows easy swapping of learners and formatters for experimentation with different anomaly detection algorithms.


## 🪸⁠ Deep Dive 🪼

TADkit includes a range of **introductory and example notebooks** that are good entry points to understand the proposed features:
- [Univariate anomaly detection example](examples/highlights/unidim_ad_example.ipynb)
  Learn how to craft your own anomaly detection method for a univariate timeseries.
- [Interactive anomaly detector demo](examples/highlights/interactive_ad_demo.ipynb)
  Experiment with multiple anomaly detectors concurrently.

### TADkit data ingestion

The `Formatter` abstract class provides array-agnostic interface for connecting your data to your anomaly detection algorithm.

TADkit offers a functional `RawToWideFormatter` that ingests your timeseries data, converts it to Wide Format and supports both pandas DataFrame and NumPy array outputs.

### Learning with TADkit

#### The TADLearner interface

`TADLearner` standardizes anomaly detection methods through a protocol that enforces:
- `.fit(X)`: for calibrating the model,
- `.score_samples(X)`: for producing anomaly scores (unbounded),
- `.predict(X)`: for producing anomaly labels (1 = normal, -1 = abnormal)

#### Catalog of methods

TADkit provides a catalog of methods enforcing the `TADLearner` interface, including the methods from the [Confiance.ai](https://www.confiance.ai/) program:
- **CNNDRAD**: two-step deep 1D-CNN for anomaly detection (representation learning + reconstruction score) - [Catalog page](https://catalog.confiance.ai/records/af2ab-hw426)
- **TDAAD**: topological data embedding + minimum covariance determinant analysis [Catalog page](https://catalog.trustworthy-ai-association.eu/records/dkvhy-nk328) and [github](https://github.com/IRT-SystemX/tdaad)
- **KCPD**: Kernel Change Point analysis for anomalies - [Catalog page](https://catalog.trustworthy-ai-association.eu/records/x3vpy-r3587) and [github](https://github.com/etaia/kernel-change-point-detection)
- **SBAD**: counterfactual-based multivariate anomaly detection and diagnosis - [Catalog page](https://catalog.confiance.ai/records/npea5-hhw40)
> Access to some libraries requires Confiance.ai credentials.

The TADkit catalog also includes base learners such as Kernel density-based anomaly detection, Gaussian mixtures anomaly detection, etc...


## 📚 Documentation & Resources

- [📖 Full API Documentation](https://irt-systemx.github.io/tadkit-core/)
- [🧪 Examples](examples/)
- [🛠 Contributing Guide](CONTRIBUTING.md)
- [🗒 Changelog](CHANGELOG.md)



## Document generation

To regenerate the documentation, rerun the following commands from the project root, adapting if
necessary:

```
pip install -r docs/docs_requirements.txt -r requirements.txt
sphinx-apidoc -o docs/source/generated tadkit
sphinx-build -M html docs/source docs/build -W --keep-going
```


## Contributors and Support

<p align="center">
  Tadkit-core is developed by
  <a href="https://www.irt-systemx.fr/en/" title="IRT SystemX">
   <img src="https://www.irt-systemx.fr/wp-content/themes/systemx/assets/medias/logo-systemx.svg" height="70">
  </a>and supported by the
<a href="https://www.trustworthy-ai-foundation.eu/" title="European Trustworthy AI association">
<img src="https://www.trustworthy-ai-association.eu/wp-content/uploads/2025/07/cropped-M0302_LOGO-ETAIA_BLANC_2000px-1-300x100.png"  height="90">
</a>
</p>
