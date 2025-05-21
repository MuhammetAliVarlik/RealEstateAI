import streamlit as st
import requests
import json

st.title("🧾 Veri İnceleme")

columns = st.text_input("Görüntülenecek Kolonlar (virgül ile):")
filter_condition = st.text_input("Filtre Koşulu (örn: Price > 500000):")
sort_by = st.text_input("Sıralama Kolonu (örn: Price):")
ascending = st.checkbox("Artan Sıralama", value=True)
group_by = st.text_input("Gruplama Kolonları (virgül ile):")
aggregations = st.text_area("Aggrege Fonksiyonlar (örn: {\"Price\": \"mean\"}):")
show_head = st.number_input("Head", min_value=0, value=0)
show_tail = st.number_input("Tail", min_value=0, value=0)

if st.button("Veriyi Göster"):
    params = {
        "show_columns": columns.split(",") if columns else None,
        "filter_condition": filter_condition or None,
        "sort_by": sort_by or None,
        "ascending": ascending,
        "group_by": group_by.split(",") if group_by else None,
        "aggregations": aggregations or None,
        "show_head": int(show_head) if show_head else None,
        "show_tail": int(show_tail) if show_tail else None
    }

    response = requests.get("http://fastapi_service:8000/dataframe_check", params=params)
    st.write("📄 Çıktı:")
    st.json(response.json())