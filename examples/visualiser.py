from typing import Any, Dict

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import streamlit as st

from tadkit.catalog.formalizers import PandasFormalizer
from tadkit.catalog.learners import installed_learner_classes
from tadkit.utils.match_formalizer_learners import match_formalizer_learners


# --- Global Settings ---
pd.options.plotting.backend = "plotly"
st.set_page_config(layout="wide")
st.title("TADkit Anomaly Detection App")

# --- Session Initialization ---
default_session_vars = [
    "uploaded_filename",
    "dataset",
    "X",
    "y",
    "formalizer",
    "matching_available_learners",
    "learners",
    "scaled_anomalies",
    "stage",
]


for var in default_session_vars:
    st.session_state.setdefault(var, None)

if st.session_state.stage is None:
    st.session_state.stage = 1


# --- Helper Functions ---
def set_stage(i: int):
    st.session_state.stage = i


def widget_factory(value_type: str):
    return {
        "range": _create_range_widget,
        "real_range": _create_range_widget,
        "choice": _create_choice_widget,
        "boolean": _create_bool_choice_widget,
    }.get(value_type, _create_bool_choice_widget)


def parameter_widget_selection(
    name: str, params_description: Dict[str, Any]
) -> Dict[str, Any]:
    with st.expander(name):
        return {
            param_name: widget_factory(desc["value_type"])(param_name, desc)
            for param_name, desc in params_description.items()
        }


def _create_range_widget(param_name: str, desc: Dict[str, Any]):
    return st.slider(
        label=param_name,
        value=desc.get("default"),
        min_value=desc.get("start"),
        max_value=desc.get("stop"),
        step=desc.get("step"),
        format="%i"
        if isinstance(desc.get("step"), int)
        else f"%0.{len(str(desc.get('step')).split('.')[-1])}f",
    )


def _create_choice_widget(param_name: str, desc: Dict[str, Any]):
    return st.radio(label=param_name, options=desc.get("set"))


def _create_bool_choice_widget(param_name: str, desc: Dict[str, Any]):
    return st.toggle(param_name)


@st.cache_data
def compute_anomalies():
    X_formalized = st.session_state.formalizer.formalize()
    anomalies = pd.DataFrame(index=st.X.index)

    for name, learner in st.session_state.learners.items():
        st.write(f"Fitting learner: {name}")
        learner.fit(X_formalized)

    for name, learner in st.session_state.learners.items():
        st.write(f"Scoring with learner: {name}")
        scores = learner.score_samples(X_formalized)
        anomalies[str(learner)] = scores

    return anomalies


@st.cache_data
def convert_for_download(df: pd.DataFrame):
    return df.to_csv().encode("utf-8")


if __name__ == "__main__":
    # --- Layout ---
    col_main, col_sidebar = st.columns([3, 1])
    upload_container = col_main.empty().container()

    uploaded_file = col_main.file_uploader("Upload a CSV file", type=["csv"])

    # --- Stage 1: Wait for File Upload ---
    if uploaded_file:
        current_filename = uploaded_file.name
        # If new file uploaded or changed
        if st.session_state.get("uploaded_filename") != current_filename:
            st.session_state.uploaded_filename = current_filename
            st.session_state.dataset = pd.read_csv(uploaded_file, header=0, index_col=0)
            st.session_state.X = st.session_state.dataset.iloc[:, :-1]
            st.session_state.y = -st.session_state.dataset.iloc[
                :, -1
            ]  # Negate if needed
            set_stage(2)
    elif st.session_state.uploaded_filename is not None:
        # File was removed
        for var in default_session_vars:
            if var != "stage":
                st.session_state[var] = None
        st.session_state.stage = 1
        st.rerun()

    # --- Stage 2: Select Learners & Parameters ---
    if st.session_state.stage >= 2:
        # Plot original time series
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        for col in st.session_state.X.columns:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.X.index,
                    y=st.session_state.X[col],
                    mode="lines",
                    name=col,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        with col_sidebar:
            col1, col2 = st.columns(2)
            if col1.button("Start Learning", use_container_width=True):
                set_stage(3)
            if col2.button("Reset", use_container_width=True):
                compute_anomalies.clear()
                set_stage(2)

            st.session_state.formalizer = PandasFormalizer(
                st.session_state.X, "synchronous"
            )
            st.session_state.formalizer.formalize()

            st.session_state.matching_available_learners = match_formalizer_learners(
                st.session_state.formalizer, installed_learner_classes
            )

            options = st.multiselect(
                "Select Anomaly Detectors:",
                options=list(st.session_state.matching_available_learners.keys()),
                default=list(st.session_state.matching_available_learners.keys()),
            )

            st.session_state.learners = {}
            for learner_name in options:
                learner_cls = st.session_state.matching_available_learners[learner_name]
                params = parameter_widget_selection(
                    name=learner_name, params_description=learner_cls.params_description
                )
                st.session_state.learners[learner_name] = learner_cls(**params)

    # --- Stage 3: Compute and Display Anomalies ---
    if st.session_state.stage >= 3:
        with col_sidebar:
            with st.status("Running anomaly detection...", expanded=True):
                anomalies = compute_anomalies()
                normalized = anomalies.apply(
                    lambda x: (x - x.min()) / (x.max() - x.min())
                )
                st.session_state.scaled_anomalies = pd.concat(
                    [normalized, st.session_state.y], axis=1
                )
                st.success("Anomaly detection complete!")

        for col in st.session_state.scaled_anomalies.columns:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.scaled_anomalies.index,
                    y=st.session_state.scaled_anomalies[col],
                    mode="lines",
                    name=col,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1),
            width=1000,
            height=600,
        )

        csv_data = convert_for_download(st.session_state.scaled_anomalies)
        st.download_button(
            label="📥 Download Results",
            data=csv_data,
            file_name="scaled_anomalies.csv",
            mime="text/csv",
        )

    # --- Final Plot ---
    if st.session_state.stage >= 2:
        upload_container.plotly_chart(fig, use_container_width=True)
