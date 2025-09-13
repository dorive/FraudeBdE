import streamlit as st
import requests
import os

st.set_page_config(page_title="Demo de Detección de Fraude", layout="wide")

API_URL = f"http://localhost:{os.getenv('API_PORT', '8000')}/score"
st.title("Demostración de Detección de Fraude")

monto = st.number_input("Importe de la transacción (€)", min_value=0.0, value=123.45)
tx_time = st.text_input("Momento de la transacción (ISO8601)", value="2025-01-01T12:00:00")

if st.button("Calcular probabilidad de fraude"):
    try:
        r = requests.post(API_URL, json={"amount": monto, "tx_time": tx_time}, timeout=3)
        st.success(f"Probabilidad estimada de fraude: {r.json()['fraud_probability']:.2%}")
    except Exception as e:
        st.error(f"Error al llamar a la API: {e}")
