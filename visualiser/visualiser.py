import pandas as pd

import streamlit as st

from tadkit.catalog.formalizers import PandasFormalizer
from tadkit.catalog.learners import installed_learner_classes
from tadkit.utils.match_formalizer_learners import match_formalizer_learners

from ui.widgets_from_metadata import render_widgets_from_metadata
from ui.plots import plot_raw_data, plot_double_data
from utils.session import (
    init_session_state,
    reset_session,
    set_stage,
    convert_for_download,
)
from utils.learning import (
    prepare_learner_configs,
    compute_anomalies_parallel,
)


init_session_state()

# ------------------------- Global Settings -------------------------
pd.options.plotting.backend = "plotly"
st.set_page_config(layout="wide")
st.title("TADkit Timeseries App")


# ------------------------- Main App -------------------------
def main():
    plot_container = st.empty().container()
    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)

    # ---------- Stage 1: Wait for File Upload ----------
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
        reset_session()
        st.rerun()

    # ---------- Stage 2: Learner Configuration ----------
    if st.session_state.stage >= 2 and st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        data = dataset.iloc[:, :-1]
        target = -dataset.iloc[:, -1]

        # Plot original data
        fig = plot_raw_data(data)

        # ---- Sidebar: Learner Controls ----
        with st.sidebar:
            st.markdown("## Learner(s) Setup")

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
            if not selected:
                st.warning("Please select at least one detector to proceed.")
                return
            if selected != st.session_state.selected_learners:
                st.session_state.selected_learners = selected
                set_stage(2)

            # Params for each learner
            learner_params = st.session_state.learner_params or {}
            for learner_name in selected:
                learner_class = available_learners[learner_name]
                learner_params.setdefault(learner_name, {})
                with st.expander(learner_name):
                    new_params = render_widgets_from_metadata(learner_class.metadata)
                learner_params[learner_name] = new_params
            st.session_state.learner_params = learner_params

        # ---------- Stage 3: Run Learners + Display ----------
        if st.session_state.stage >= 3:
            with st.sidebar:
                st.markdown("## Learning")
                learner_configs, cache_key = prepare_learner_configs(
                    selected, learner_params, available_learners
                )
                anomalies, predictions, errors = compute_anomalies_parallel(
                    X=data,
                    learner_configs=learner_configs,
                    cache_key=cache_key,
                )
                for err in errors:
                    st.warning(err)
                # st.dataframe(anomalies)

                result_type = st.segmented_control(
                    "Select result type to display",
                    ["Anomaly scores", "Anomalies"],
                    default="Anomaly scores",
                )
                output = anomalies if result_type == "Anomaly scores" else predictions
                output[dataset.columns[-1]] = target

            fig = plot_double_data(data, output)

            # Download Button
            st.download_button(
                label="Download anomalies CSV",
                data=convert_for_download(output),
                file_name="scaled_anomalies.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        plot_container.plotly_chart(fig, use_container_width=True)


# ------------------------- Entry Point -------------------------
if __name__ == "__main__":
    main()
