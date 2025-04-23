import streamlit as st
import pandas as pd

from pathlib import Path

from tadkit.catalog.formalizers import PandasFormalizer
from tadkit.catalog.learners import installed_learner_classes
from tadkit.utils.match_formalizer_learners import match_formalizer_learners


st.title('Dummy tadkit app')

uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
file = Path("examples/ornhul.csv")

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file, header=0, index_col=0)
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    st.subheader('Your data looks like this')
    pd.options.plotting.backend = "plotly"
    fig = pd.concat([X, y], axis=1).plot()
    fig.update_layout(
        legend=dict(
            orientation="v",
            entrywidth=100,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=1000,
        height=600,
    )
    st.plotly_chart(fig)

    st.subheader('Anomaly learning...')
    formalizer = PandasFormalizer(data_df=X, dataframe_type="synchronous")
    X_test = formalizer.formalize(**{})
    matching_available_learners = match_formalizer_learners(formalizer, installed_learner_classes)

    options = st.multiselect(
        "What detector do you want to use?",
        matching_available_learners.keys(),
        matching_available_learners.keys(),
    )
    st.write("You selected:", options)
    learners = {key: matching_available_learners[key]() for key in options}

    # Fit loop:
    anomalies = pd.DataFrame(index=X.index)
    for learner_name, learner_object in learners.items():
        print(f"Fitting {learner_name}")
        learner_object.fit(X_test)

    # Score loop:
    for learner_name, fitted_learner_object in learners.items():
        print(f"Scoring with {learner_name}")
        anom_score = fitted_learner_object.score_samples(X_test)
        anomalies[str(fitted_learner_object)] = anom_score

    st.subheader('Your anomalies analysis:')
    pd.options.plotting.backend = "plotly"
    fig = pd.concat([anomalies.apply(lambda x: (x - x.min()) / (x.max() - x.min())), y], axis=1).plot()
    fig.update_layout(
        legend=dict(
            orientation="v",
            entrywidth=100,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=1000,
        height=600,
    )
    st.plotly_chart(fig)
