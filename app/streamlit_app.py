"""Local application placeholder, not a functioning comparison UI."""

import streamlit as st

st.title("DeskCheck")
st.warning("Skeleton only: comparison pipeline and trained checkpoint are not implemented.")
st.file_uploader("Before image", type=["jpg", "jpeg", "png"], key="before")
st.file_uploader("After image", type=["jpg", "jpeg", "png"], key="after")
st.button("Compare", disabled=True)
