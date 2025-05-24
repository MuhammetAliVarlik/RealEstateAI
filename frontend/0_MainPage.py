import requests
import streamlit as st
import json

FASTAPI_STREAM_URL = "http://fastapi_service:8000/stream"

st.set_page_config(page_title="Real Estate App", page_icon="🏠")

st.title("🏡 Real Estate Analysis and Chatbot App")
st.markdown("""
This application offers the following features:

- 🤖 **AI Chatbot** — Question-answer AI assistant  
- 💰 **House Price Prediction** — Predict house price based on features  
- 🚨 **Anomaly Detection** — Detect price anomalies  
- 🏷️ **House Type Prediction** — Automatically predict house type  
- 💼 **Investment Analysis** — Check investment potential  
- 📊 **Dataset Exploration** — Analyze your dataset  

Use the menu in the top left to select a page.
""")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Mesajınızı yazın..."):
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_box = st.chat_message("assistant")
    placeholder = response_box.empty()
    placeholder.markdown("⏳ Waiting for response...")

    collected = ""
    with requests.get(FASTAPI_STREAM_URL, params={"question": prompt}, stream=True) as response:
        for line in response.iter_lines():
            if line:
                part = line.decode("utf-8")
                collected += part
                try:
                    parsed = json.loads(collected)
                    placeholder.markdown(parsed.get("response", collected))
                except json.JSONDecodeError:
                    placeholder.markdown(collected)

    try:
        parsed = json.loads(collected)
        final_response = parsed.get("response", collected)
    except json.JSONDecodeError:
        final_response = collected

    st.session_state["messages"].append({"role": "assistant", "content": final_response})
