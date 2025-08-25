import streamlit as st

st.title("This is my Python Webpage")
############## LEARNING SECTION ##############

# Explanation detailed here https://grok.com/share/bGVnYWN5_0467c5ab-e00c-4d7b-b2f4-687da7b81b4f

# st.write("This is normal text.")
# st.markdown("**This is bold text!**")  # Supports bold, italic, etc.
# st.header("This is a smaller heading")

# name = st.text_input(label="Username")
# if name:
#     st.write(f"Hi, {name}!")



# Initialize a place to store messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Show all saved messages
# for message in st.session_state.messages:
#     st.write(f"{message['role']}: {message['content']}")

# # Get user input
# user_input = st.text_input("Type a message:")

# # If the user types something, save and display it
# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     st.write(f"user: {user_input}")



############## LEARNING SECTION ##############

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
