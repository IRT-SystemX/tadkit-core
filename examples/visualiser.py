
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
from skrub import TableReport

from tadkit.catalog.formalizers import PandasFormalizer
from tadkit.catalog.learners import installed_learner_classes
from tadkit.utils.match_formalizer_learners import match_formalizer_learners


from typing import Any, Dict


@st.cache_data
def set_state(i):
    st.session_state.stage = i


def widget_matcher(value_type):
    match value_type:
        case "range" | "real_range":
            return _create_range_widget
        case "choice":
            return _create_choice_widget
        case "boolean":
            return _create_bool_choice_widget
    return _create_bool_choice_widget


def parameter_widget_selection(name: str, params_description: Dict[str, Any]) -> Dict[str, Any]:
    with st.expander(name):
        params = {}
        for param_name, param_description in params_description.items():
            creator = widget_matcher(param_description["value_type"])
            params[param_name] = creator(param_name, param_description)
    return params


def _create_range_widget(param_name: str, param_description: Dict[str, Any]):
    value = param_description.get("default")
    min_value = param_description.get("start")
    max_value = param_description.get("stop")
    step = param_description.get("step")

    myformat = "%i" if isinstance(step, int) else f"%0.{len(str(step).replace('.', ''))}f"

    return st.slider(
        label=param_name,
        value=value,
        min_value=min_value,
        max_value=max_value,
        step=step,
        format=myformat,
    )


def _create_choice_widget(param_name: str, param_description: Dict[str, Any]):
    choice_widget = st.radio(
        label=param_name,
        options=param_description.get("set"),
    )
    return choice_widget


def _create_bool_choice_widget(param_name: str, param_description: Dict[str, Any]):
    bool_choice_widget = st.toggle(param_name)
    return bool_choice_widget


@st.cache_data
def compute_anomalies():
    X_test = st.session_state.formalizer.formalize(**{})

    # Fit loop:
    anomalies = pd.DataFrame(index=st.X.index)
    for learner_name, learner_object in st.session_state.learners.items():
        st.write(f"Anomaly learning -- fitting {learner_name}")
        learner_object.fit(X_test)

    # Score loop:
    for learner_name, fitted_learner_object in st.session_state.learners.items():
        st.write(f"Anomaly learning -- scoring with {learner_name}")
        anom_score = fitted_learner_object.score_samples(X_test)
        anomalies[str(fitted_learner_object)] = anom_score
    return anomalies


@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")


st.set_page_config(layout="wide")

st.title('Dummy tadkit app')


col1, col2 = st.columns([3, 1])
# file = Path("examples/ornhul.csv")

if 'stage' not in st.session_state:
    st.session_state.stage = 1


if st.session_state.stage >= 1:
    uploaded_file = col1.file_uploader("Choose a CSV file", accept_multiple_files=False)
    if uploaded_file is not None:
        st.session_state.dataset = pd.read_csv(uploaded_file, header=0, index_col=0)
        st.X, st.y = st.session_state.dataset.iloc[:, :-1], -st.session_state.dataset.iloc[:, -1]
        # pd.options.plotting.backend = "plotly"
        fig = st.X.plot()
        fig.update_layout(
            legend=dict(
                orientation="v",
                entrywidth=100,
                yanchor="auto",
                y=1.02,
                xanchor="auto",
                x=1
            ),
        )
        col1.plotly_chart(fig)
        set_state(i=2)

if st.session_state.stage >= 2:
    with col2:
        st.button("Start learning", icon="🌊", on_click=set_state, args=[3])
        st.session_state.formalizer = PandasFormalizer(data_df=st.X, dataframe_type="synchronous")
        st.session_state.formalizer.formalize(**{})
        st.session_state.matching_available_learners = match_formalizer_learners(
            st.session_state.formalizer, installed_learner_classes)

        options = st.multiselect(
            "Choose detectors for anomaly learning:",
            st.session_state.matching_available_learners.keys(),
            st.session_state.matching_available_learners.keys(),
        )

        st.session_state.learners = {}
        for option in options:
            learner = st.session_state.matching_available_learners[option]
            params = parameter_widget_selection(name=option, params_description=learner.params_description)
            st.session_state.learners[option] = learner(**params)


if st.session_state.stage >= 3:
    with col2:
        with st.status("Anomaly learning...") as status:
            anomalies = compute_anomalies()
            st.session_state.scaled_anomalies = pd.concat(
                [anomalies.apply(lambda x: (x - x.min()) / (x.max() - x.min())), st.y], axis=1)
            st.write("Anomaly learning -- done.")
            set_state(i=4)


if st.session_state.stage >= 4:

    st.subheader("Data anomaly scores")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    data = st.X
    for col in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col, showlegend=False), row=1, col=1)

    for col in st.session_state.scaled_anomalies.columns:
        fig.add_trace(go.Scatter(x=st.session_state.scaled_anomalies.index,
                                 y=st.session_state.scaled_anomalies[col], mode='lines', name=col), row=2, col=1)

    fig.update_layout(
        legend=dict(
            orientation="v",
            entrywidth=100,
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=1000,
        height=600,
    )
    st.plotly_chart(fig)

    csv = convert_for_download(st.session_state.scaled_anomalies)

    st.download_button(
        label="Download anomalies CSV",
        data=csv,
        file_name="scaled_anomalies.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.write("Skrub scaled anomalies report:")
    components.html(TableReport(st.session_state.scaled_anomalies).html_snippet(), height=1000)
