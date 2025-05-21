import streamlit as st
import requests

st.title("🏡 Ev Tipi Tahmini")

params = {
    "Price": st.number_input("Fiyat", value=800000.0),
    "GrossSquareMeters": st.number_input("Brüt Metrekare", value=120.0),
    "ItemStatus": st.selectbox("İlan Durumu", ["Eşyalı", "Boş"]),
    "room": st.number_input("Oda Sayısı", value=3.0),
    "hall": st.number_input("Salon Sayısı", value=1.0),
    "district": st.text_input("Semt / İlçe", value="Ataşehir")
}

if st.button("Tahmin Et"):
    response = requests.get("http://fastapi_service:8000/home_type", params=params)
    st.write("🏷️ Tahmini Tip:")
    st.json(response.json())