import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import streamlit as st

from tadkit.catalog.formalizers import PandasFormalizer
from tadkit.catalog.learners import installed_learner_classes
from tadkit.utils.match_formalizer_learners import match_formalizer_learners

from ui.widgets import parameter_widget_selection
from utils.session import set_stage, convert_for_download
from utils.learning import (
    prepare_learner_configs,
    compute_anomalies_parallel,
)


# ------------------------- Global Settings -------------------------
pd.options.plotting.backend = "plotly"
st.set_page_config(layout="wide")
st.title("TADkit Timeseries App")

# ------------------------- Session Initialization -------------------------
DEFAULTS = {
    "stage": 1,
    "uploaded_filename": None,
    "dataset": None,
    "selected_learners": None,
    "learner_params": None,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ------------------------- Main App -------------------------
def main():
    plot_container = st.empty().container()
    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)

    # --- Stage 1: Wait for File Upload ---
    if uploaded_file:
        current_filename = uploaded_file.name
        # If new file uploaded or changed
        if st.session_state.uploaded_filename != current_filename:
            st.session_state.uploaded_filename = current_filename
            df = pd.read_csv(uploaded_file, index_col=0)
            st.session_state.dataset = df
            st.session_state.X = df.iloc[:, :-1]
            st.session_state.y = -df.iloc[:, -1]  # Adjust sign if needed
            set_stage(2)

    elif st.session_state.uploaded_filename is not None:
        st.session_state.update(DEFAULTS)
        st.rerun()

    # ---------- Stage 2: Learner Configuration ----------
    if st.session_state.stage >= 2 and st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        data = dataset.iloc[:, :-1]
        target = -dataset.iloc[:, -1]

        # Plot original data
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        for col in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data[col], mode="lines", showlegend=False),
                row=1,
                col=1,
            )

        # ---- Sidebar: Learner Controls ----
        with st.sidebar:
            st.markdown("## Learner Setup")

            col1, col2 = st.columns(2)
            if col1.button("Start learning", type="primary"):
                set_stage(3)

            if col2.button("Reset learning"):
                compute_anomalies_parallel.clear()
                set_stage(2)

            formalizer = PandasFormalizer(data_df=data, dataframe_type="synchronous")
            formalizer.formalize()
            available_learners = match_formalizer_learners(
                formalizer, installed_learner_classes
            )

            selected = st.multiselect(
                "Choose detectors:",
                list(available_learners.keys()),
                st.session_state.selected_learners or list(available_learners.keys()),
            )
            if selected != st.session_state.selected_learners:
                set_stage(2)
            st.session_state.selected_learners = selected

            # Params for each learner
            learner_params = st.session_state.learner_params or {}
            for learner_name in selected:
                learner_class = available_learners[learner_name]
                param_desc = learner_class.params_description
                learner_params.setdefault(learner_name, {})
                new_params = parameter_widget_selection(learner_name, param_desc)
                learner_params[learner_name] = new_params
            st.session_state.learner_params = learner_params

        # ---------- Stage 3: Run Learners + Display ----------
        if st.session_state.stage >= 3:
            with st.sidebar.status("Learning in progress..."):
                learner_configs, cache_key = prepare_learner_configs(
                    selected, learner_params, available_learners
                )
                anomalies = compute_anomalies_parallel(data, learner_configs, cache_key)

            # Normalize and include ground truth
            scaled = anomalies.apply(
                lambda col: (col - col.min()) / (col.max() - col.min())
            )
            scaled[dataset.columns[-1]] = target

            for col in scaled.columns:
                fig.add_trace(
                    go.Scatter(
                        x=scaled.index,
                        y=scaled[col],
                        mode="lines",
                        name=col,
                    ),
                    row=2,
                    col=1,
                )

            fig.update_layout(
                legend=dict(
                    orientation="v",
                    entrywidth=100,
                    yanchor="top",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                width=1000,
                height=600,
            )

            # Download Button
            st.download_button(
                label="Download anomalies CSV",
                data=convert_for_download(scaled),
                file_name="scaled_anomalies.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        plot_container.plotly_chart(fig, use_container_width=True)


# ------------------------- Entry Point -------------------------
if __name__ == "__main__":
    main()
