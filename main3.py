import streamlit as st

st.title("This is my Python Webpage")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # User or Assistant/Bot
        st.markdown(message["content"])

if user_input := st.chat_input("Type Something..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
