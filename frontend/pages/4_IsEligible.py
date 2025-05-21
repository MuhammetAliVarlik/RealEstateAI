import streamlit as st
import requests

st.title("💼 Yatırım Uygunluğu")

params = {
    "Price": st.number_input("Fiyat", value=950000.0),
    "GrossSquareMeters": st.number_input("Brüt Metrekare", value=140.0),
    "ItemStatus": st.selectbox("İlan Durumu", ["Eşyalı", "Boş"]),
    "room": st.number_input("Oda Sayısı", value=3.0),
    "hall": st.number_input("Salon Sayısı", value=1.0),
    "district": st.text_input("Semt / İlçe", value="Maltepe")
}

if st.button("Uygun mu?"):
    response = requests.get("http://fastapi_service:8000/is_eligible", params=params)
    st.write("📈 Uygunluk Durumu:")
    st.json(response.json())