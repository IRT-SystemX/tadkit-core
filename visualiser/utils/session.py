import streamlit as st
import pandas as pd


def set_stage(i: int):
    st.session_state.stage = i


@st.cache_data
def convert_for_download(df: pd.DataFrame):
    return df.to_csv().encode("utf-8")
