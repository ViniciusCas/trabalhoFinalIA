import os 
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler 

numerical_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'credit_score']
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']

rf_model = joblib.load("models/RF_model.pkl")
rl_model = joblib.load("models/RL_model.pkl")

le = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

if 'final_data' not in st.session_state:
    st.session_state.final_data = False

def treath_csv_data(csv_file):
    if csv_file is None:
        st.error("Por favor, insira um arquivo csv com os dados dos clientes.")
        return
    
    batch_data = pd.read_csv(csv_file)
    
    if batch_data.empty:
        st.error("O arquivo csv está vazio.")
        return
    
    if batch_data.columns.tolist() != ['person_age', 'person_gender', 'person_education', 'person_income',
                                'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
                                'loan_int_rate', 'cb_person_cred_hist_length',
                                'credit_score', 'previous_loan_defaults_on_file']:
        st.error("O arquivo csv não segue o template correto.")
        return
    
    batch_data['loan_percent_income'] = batch_data['loan_amnt'] / batch_data['person_income'].replace(0, np.nan)
    
    batch_data[categorical_cols] = le.transform(batch_data[categorical_cols])

    batch_data[numerical_cols] = scaler.fit_transform(batch_data[numerical_cols])

    st.session_state.final_data = True
    return batch_data

def treath_data(data):
    if data is None:
        st.error("Por favor, insira os dados do cliente.")
        return

    data['loan_percent_income'] = data['loan_amnt'] / data['person_income'].replace(0, np.nan)

    for i in data:
        st.write(i)
    for rows in data.iterrows():
        st.write(rows)

    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
        
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    st.session_state.final_data = True
    return data

st.subheader("Selecione um modelo para classificar")

model_selection = st.radio(
    "Selecione o modelo", 
    ["Regressão Logística", "Random Forest"],
    horizontal=True,
)

if model_selection == "Regressão Logística":
    model = rl_model
else:
    model = rf_model

uploaded_file = None


oneEntry, batchEntry = st.tabs(["Entrada única", "Entrada em lotes"])

with oneEntry:
    st.text("Insira os dados do cliente para classificação")

    person_age = float(st.number_input("Idade", min_value=18.0, max_value=150.0, step=0.1))
    person_gender = st.selectbox("Sexo", ["male", "female"])
    person_education = st.selectbox("Nível de educação", 
                                    ["Associate", "High School", "Bachelor",
                                      "Master", "Doctorate"])
    person_income = float(st.number_input("Renda anual", min_value=0.0, step=0.1))
    person_emp_exp = st.number_input("Anos de experiência profissional", min_value=0)
    person_home_ownership = st.selectbox("Situação como proprietário", 
                                         ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = float(st.number_input("Valor do empréstimo", min_value=0.0, step=0.1))
    loan_intent = st.selectbox("Propósito do empréstimo",
                                ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE',
                                  'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_int_rate = float(st.number_input("Taxa de juros do empréstimo", min_value=0.0, max_value=100.0, step=0.001))
    cb_person_cred_hist_length = float(st.number_input("Comprimento do histórico de crédito em anos", min_value=0.0, step=0.001))
    credit_score = st.number_input("Score de crédito", min_value=0)
    previous_loan_defaults_on_file = st.selectbox("Indicador de inadimplência de empréstimos anteriores", 
                                                       ["No", "Yes"])
    
    data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [np.nan],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })

    if st.columns([2,1,2])[1].button("Classificar", use_container_width=True, key="uniqueBtn"):
        final_data = treath_data(data)

with batchEntry:
    uploaded_file = st.file_uploader("Anexe um arquivo csv com os dados dos clientes para classificação",
                                      type="csv")

    st.text("O template abaixo deve ser seguido:")

    st.write(pd.DataFrame({
        'person_age': [],
        'person_gender': [],
        'person_education': [],
        'person_income': [],
        'person_emp_exp': [],
        'person_home_ownership': [],
        'loan_amnt': [],
        'loan_intent': [],
        'loan_int_rate': [],
        'cb_person_cred_hist_length': [],
        'credit_score': [],
        'previous_loan_defaults_on_file': []
    }))

    if st.columns([2,1,2])[1].button("Classificar", use_container_width=True, key="batchBtn"):
        final_data = treath_csv_data(uploaded_file)


if st.session_state.final_data:
    st.subheader("Resultado da classificação")
    
    input_features = np.array(final_data)
    prediction = model.predict(input_features)

    for index, status in enumerate(prediction):
        st.write(prediction[index])
        if status == 1:
            st.success(f"Pessoa {index} aprovado para o empréstimo")
        else:
            st.warning(f"Pessoa {index} rejeitado para o empréstimo")

    st.session_state.final_data = False