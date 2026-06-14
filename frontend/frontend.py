import streamlit as st
import requests
import uuid
import time
from pathlib import Path

API_URL = "http://rag-api:8000/trigger_rag"

st.set_page_config(
    page_title="ShopEasy Support Bot",
    page_icon="🛍️",
    layout="centered"
)

IMAGE_PATH = Path("src/images/customer_support.png")

if IMAGE_PATH.exists():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(str(IMAGE_PATH), use_container_width=True)

st.title("ShopEasy Customer Support")
st.caption("Ask your questions about ShopEasy policies.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def fake_stream(text: str):
    for char in text:
        yield char
        time.sleep(0.005)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_query,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    answer = response.json().get("response", "No response received.")
                else:
                    answer = f"Error: API returned status code {response.status_code}"

        except Exception as e:
            answer = f"Could not connect to backend: {e}"

        streamed_answer = st.write_stream(fake_stream(answer))

    st.session_state.messages.append({
        "role": "assistant",
        "content": streamed_answer
    })