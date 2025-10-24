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
  - `Formalizer`: prepares raw data into a machine-learning-friendly format.
  - `TADLearner`: implements `.fit(X)`, `.score_samples(X)`, and `.predict(X)` for unsupervised anomaly detection.

- **Supports Multiple Detection Methods**
  Includes methods from scikit-learn and Confiance.ai components ([TDAAD](https://catalog.confiance.ai/records/xvc80-whm36) and [github](https://github.com/IRT-SystemX/tdaad), [SBAD](https://catalog.confiance.ai/records/fkpja-s7546), [KCPD](https://catalog.confiance.ai/records/kxc1c-12x55) and [github](https://github.com/confianceai/kernel-change-point-detection), [CNNDRAD](https://catalog.confiance.ai/records/af2ab-hw426), ...). All learners can be instantiated with default parameters or customized as needed.

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
from tadkit.formalizers import Formalizer

# Prepare your data
formalizer = Formalizer()
X = formalizer.fit_transform(raw_data)

# Initialize and fit the learner
# @TODO
for name, learner in formalizer.available_learners():
    # Learner calibration
    learner.fit(X)
    # Anomaly scores
    y_score = learner.score_samples(X)
    # Detect anomalies
    predictions = learner.predict(X)
```

The modular architecture allows easy swapping of learners and formalizers for experimentation with different anomaly detection algorithms.


## 🪸⁠ Deep Dive 🪼

`tadkit-core` also provides a range of **introductory and example notebooks** to help you explore the interfaces and methods:

- [Univariate anomaly detection example](examples/highlights/unidim_ad_example.ipynb)
  Learn how to craft your own anomaly detection method for a univariate timeseries.
- [Interactive anomaly detector demo](examples/highlights/interactive_ad_demo.ipynb)
  Experiment with multiple anomaly detectors concurrently.

### TADkit Formalizer

The `Formalizer` abstract class bridges **raw data and anomaly detection methods**.
- **PandasFormalizer**: a ready-to-use formalizer for basic tasks.
- **Custom Formalizers**: required for complex tasks or when a method requires specific data formatting.

**Key properties:**
- `available_properties`: list of tags for automatic matching with compatible `TADLearner`s.
- `query_description`: describes parameters of the `formalize` method:

    {
        <param_name>: {
            'description': <str describing the parameter>,
            'family': <tag like 'time', 'space', 'preprocessing'>,
            'value_type': <type tag like 'interval_element', 'set_element', 'subset'>,
            ...
        },
        ...
    }

The `formalize(query)` method returns the corresponding query data.

### TADLearner Interface

`TADLearner` standardizes anomaly detection methods.
Required methods:
- `.fit(X)`: calibrate the model
- `.score_samples(X)`: produce anomaly scores
- `.predict(X)`: produce anomaly labels (1 = normal, -1 = abnormal)

**Key attributes:**
- `required_properties`: ensures compatibility with `Formalizer.available_properties`.
- `params_description`: dictionary describing model parameters.

### Integrated Libraries

`TADLearner` format includes Confiance.ai and standard methods:

- **CNNDRAD**: two-step deep 1D-CNN for anomaly detection (representation learning + reconstruction score) - [catalog.confiance.ai/records/af2ab-hw426](https://catalog.confiance.ai/records/af2ab-hw426)
- **TDAAD**: topological data embedding + minimum covariance determinant analysis [catalog.confiance.ai/records/ve158-h4h60](https://catalog.confiance.ai/records/ve158-h4h60) and [github](https://github.com/IRT-SystemX/tdaad)
- **KCPD**: Kernel Change Point analysis for anomalies - [catalog.confiance.ai/records/6atzy-3yn05](https://catalog.confiance.ai/records/6atzy-3yn05) and [github](https://github.com/confianceai/kernel-change-point-detection)
- **SBAD**: counterfactual-based multivariate anomaly detection and diagnosis - [catalog.confiance.ai/records/npea5-hhw40](https://catalog.confiance.ai/records/npea5-hhw40)

> Access to some libraries requires Confiance.ai credentials.

### Creating Your Own TADLearner

Tools for custom learners:
- `sklearn_tadlearner_factory`: wraps a scikit-learn model into a `TADLearner`
- `decomposable_tadlearner_factory`: creates a learner pipeline from a preprocessor + learner


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
   <img src="https://www.irt-systemx.fr/wp-content/uploads/2013/03/system-x-logo.jpeg"  height="70">
  </a>and supported by the
<a href="https://www.trustworthy-ai-foundation.eu/" title="European Trustworthy AI association">
<img src="https://www.trustworthy-ai-association.eu/wp-content/uploads/2025/07/cropped-M0302_LOGO-ETAIA_BLANC_2000px-1-300x100.png"  height="90">
</a>
</p>
