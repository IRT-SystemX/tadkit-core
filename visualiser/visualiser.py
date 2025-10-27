import pandas as pd

import streamlit as st

from tadkit.catalog.rawtowideformatter import RawToWideFormatter
from tadkit.utils.ui import render_widgets_from_params
from tadkit.utils.param_spec import params_from_class

from tadkit.base.registry import registry
import tadkit.catalog.registry_init  # ensure registrations happen


from plots import plot_raw_data, plot_double_data
from session import (
    init_session_state,
    reset_session,
    set_stage,
    convert_for_download,
)
from learning import (
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
    if tadkit.catalog.registry_init:
        pass

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

            formatter = RawToWideFormatter(data=data, backend="pandas")
            formatter.format()

            # registry.print_catalog_classes()
            # registry.list_learners()

            available_learners = registry.match_learners(formatter)
            learners_select = {
                learner.__name__: learner for learner in available_learners
            }

            selected = st.multiselect(
                "Choose detectors:",
                learners_select.keys(),
                st.session_state.selected_learners or learners_select.keys(),
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
                learner_class = learners_select[learner_name]
                learner_params.setdefault(learner_name, {})
                with st.expander(learner_name):
                    param_specs = params_from_class(learner_class)
                    new_params_widget, new_params = render_widgets_from_params(
                        param_specs, frontend="st"
                    )
                learner_params[learner_name] = new_params
            st.session_state.learner_params = learner_params

        # ---------- Stage 3: Run Learners + Display ----------
        if st.session_state.stage >= 3:
            with st.sidebar:
                st.markdown("## Learning")

                learner_configs, cache_key = prepare_learner_configs(
                    [
                        learners_select[learner_name]
                        for learner_name in st.session_state.selected_learners
                    ],
                    learner_params,
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
