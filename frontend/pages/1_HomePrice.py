import streamlit as st
import requests

st.title("🏠 Ev Fiyat Tahmini")

params = {
    "GrossSquareMeters": st.number_input("Brüt Metrekare", value=100.0),
    "ItemStatus": st.selectbox("İlan Durumu", ["Boş", "Eşyalı"]),
    "room": st.number_input("Oda Sayısı", value=2.0),
    "hall": st.number_input("Salon Sayısı", value=1.0),
    "district": st.text_input("Semt / İlçe", value="Kadıköy")
}

if st.button("Tahmin Et"):
    response = requests.get("http://fastapi_service:8000/home_price", params=params)
    st.write("📊 Tahmini Fiyat:")
    st.json(response.json())