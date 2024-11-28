import streamlit as st
import pandas as pd

st.header("Sobre o classificador")

st.text("""Essa é uma aplicação que visa classificar se um cliente será aprovado ou não 
        para um pedido de empréstimo.""")

st.text("""O classificador foi treinado com base em dados históricos de clientes que solicitaram 
        empréstimos e foi treinado com dois algoritmos de aprendizado de máquina: Regressão Logística 
        e Random Forest.""")

st.markdown("""Para o banco de dados foi utilizado o dataset 'Loan Prediction' disponível no 
            [Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)""")

st.header("Banco de dados")

st.dataframe(pd.DataFrame({ 
    "Coluna": ["person_age", "person_gender", "person_education", "person_income", 
               "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent", 
               "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", 
               "credit_score", "previous_loan_defaults_on_file", "loan_status"], 
    "Descrição": ["Idade da pessoa", "Sexo da pessoa", "Maior nível de educação", 
                  "Ganhos anuais", "Anos de experiência de trabalho", 
                  "Situação como proprietário", "Valor do empréstimo", 
                  "Propósito do emprestimo", "Taxa de juros do emprestimo", 
                  "Valor do empréstimo como porcentagem da renda anual", 
                  "Comprimento do histórico de crédito em anos", "Crédito da pessoa", 
                  "Indicador de inadimplência de empréstimos anteriores", 
                  "Status de aprovação do empréstimo: 1 = aprovado; 0 = rejeitado"], 
    "Tipo": ["Float", "String", "String", "Float", "Inteiro", "String", "Float", 
             "String", "Float", "Float", "Float", "Inteiro", "String", "Inteiro"]
}))

st.header("Algortimos de aprendizado de máquina")

st.text("""Para realizar a classificação, dois algoritmos de aprendizado de máquina foram
        utilizados: Regressão Logística e Random Forest. Abaixo, segue uma breve descrição de cada""")

st.subheader("Regressão Logística")

st.text("""A regressão logística é um algoritmo de aprendizado supervisionado que é usado para prever
        a probabilidade de uma variável dependente categórica. Em problemas de classificação binária,
        a regressão logística prevê a probabilidade de um evento ocorrer.""")

with st.columns([1,2,1])[1]:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTK1kAdtwqHxZOQOdNvgFc5il8tHCmq-Xduw&s", use_container_width=True)

st.subheader("Random Forest")

st.text("""Random Forest é um algoritmo de aprendizado de máquina que cria um conjunto de árvores de decisão
        durante o treinamento e fornece a classe que é o modo das classes das árvores individuais.""")

with st.columns([1,2,1])[1]:
    st.image("https://miro.medium.com/v2/resize:fit:592/1*i0o8mjFfCn-uD79-F1Cqkw.png", use_container_width=True)

