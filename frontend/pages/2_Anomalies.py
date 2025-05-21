import streamlit as st
import requests

st.title("🚨 Fiyat Anomali Tespiti")

params = {
    "Price": st.number_input("Fiyat", value=500000.0),
    "GrossSquareMeters": st.number_input("Brüt Metrekare", value=100.0),
    "ItemStatus": st.selectbox("İlan Durumu", ["Eşyalı", "Boş"]),
    "room": st.number_input("Oda Sayısı", value=2.0),
    "hall": st.number_input("Salon Sayısı", value=1.0),
    "district": st.text_input("Semt / İlçe", value="Beşiktaş")
}

if st.button("Anomali Kontrolü"):
    response = requests.get("http://fastapi_service:8000/anomalies", params=params)
    st.write("🧪 Anomali Sonucu:")
    st.json(response.json())