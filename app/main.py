import streamlit as st

about_page = st.Page("pages/about.py", title="Sobre", icon="📚")
models_page = st.Page("pages/models.py", title="Modelos", icon="🤖")

st.title("Classificador para aprovação de empréstimo")
st.divider()

pg = st.navigation([about_page, models_page])
pg.run()


