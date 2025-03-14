## Active Learning

Active learning is a powerful technique aimed at enhancing machine learning models, particularly when labeling data is expensive or time-consuming. When dealing with large unlabeled datasets (such as those used in anomaly detection), randomly selecting instances to label is often inefficient, especially since anomalies are rare. Active learning optimizes this process by strategically selecting the most informative instances to label, thereby improving the model’s performance with fewer labeled examples. This approach leads to more effective learning of both normal and anomalous patterns with less effort.

In an active learning workflow, the key elements are the models and the query strategy used to decide which data points to label next.

![al overview](/docs/source/images/al_overview.png "Active Learning Overview")

## Conformal Active Anomaly Detection

Our approach involves using multiple anomaly detection models (committe), each with its own anomaly score function. The models are calibrated to align on the same tolerance criteria, and uncertainty sampling is employed based on the associated tolerance bounds. For each model, the uncertainty score range is calculated using conformal prediction, and disagreement votes filter samples in the uncertainty region. Following this, a vote among the models determines the final decision, ensuring that the most informative and uncertain samples are selected first for labeling or further analysis.

![cal overview](/docs/source/images/cal_overview.png "Conformal Active Learning Overview")

### Algorithm Description

Consider $n$ anomaly detection models $\hat{f}_1, \dots, \hat{f}_n$, each associated with an anomaly score function $s_1, \dots, s_n$. The decision function for each model $\hat{f}_i$ is defined as:

$$
\hat{f}_i^{M_i}(x) = \begin{cases} 
\text{anomaly} & \text{if } s_i(x) > M_i, \\
\text{normal} & \text{if } s_i(x) \leq M_i,
\end{cases}
$$

where $M_i$ is the anomaly threshold for model $\hat{f}_i$.

1. **Calibration**:
   - For each Anomaly Detection (AD) model $f_i$, calibrate the threshold according to specified tolerance criteria $\alpha_1$ and $\alpha_2$, resulting in thresholds $M_i^l$ and $M_i^h$. Apply Bonferroni correction to account for multiple comparisons.

2. **Sample Selection**:
   - Select samples whose anomaly scores fall within the calibrated range $[M_i^l, M_i^h]$ for each AD model $\hat{f}_i$.

3. **Uncertainty Sampling**:
   - Conduct a disagreement vote $\hat{v}_i^{US}$ among the models on these selected samples to identify those with the highest uncertainty.
    $$
    \hat{v}_i^{US}(x) = \hat{f}_i^{M_i^l}(x) \oplus \hat{f}_i^{M_i^h}(x)
    $$

4. **Query by Committee (QBC)**:
   - Use a vote among the models to make a ranking of most uncertain samples. Most voted samples are to be labeled first.

## Getting Started 

### Installation

Make sure to install dependencies:

```bash
pip install puncc git+https://github.com/modAL-python/modAL
```

### Demo

Check [this notebook](/examples/demo_active_learning.ipynb) for a full demo on active anomaly detection for time series. 

