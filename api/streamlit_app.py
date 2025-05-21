# streamlit_app.py
import streamlit as st
import requests

st.title("🔁 FastAPI Streaming Arayüzü")

prompt = st.text_input("Prompt girin", "Merhaba, bilgi verir misin?")
start_button = st.button("Gönder")

FASTAPI_STREAM_URL = "http://localhost:8000/stream"

def stream_from_fastapi(prompt):
    with requests.get(FASTAPI_STREAM_URL, params={"prompt": prompt}, stream=True) as response:
        collected = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                part = chunk.decode('utf-8')
                collected += part
                yield collected

if start_button:
    st.info("Yanıt alınıyor...")
    placeholder = st.empty()
    for partial in stream_from_fastapi(prompt):
        placeholder.markdown(f"```\n{partial}\n```")
