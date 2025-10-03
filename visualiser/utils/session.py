# utils/session.py

import streamlit as st
import pandas as pd

SESSION_KEYS = [
    "stage",
    "uploaded_filename",
    "dataset",
    "selected_learners",
    "learner_params",
    "X",
    "y",
]


def init_session_state():
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.stage is None:
        st.session_state.stage = 1
    if st.session_state.selected_learners is None:
        st.session_state.selected_learners = []
    if st.session_state.learner_params is None:
        st.session_state.learner_params = {}


def reset_session(keep_keys=["stage"]):
    for key in SESSION_KEYS:
        if key not in keep_keys:
            st.session_state[key] = None


def set_stage(i: int):
    st.session_state.stage = i


@st.cache_data
def convert_for_download(df: pd.DataFrame):
    return df.to_csv().encode("utf-8")
