import streamlit as st

# Initialize chat history in session state
if 'store' not in st.session_state:
    st.session_state.store = {'chat_history': []}

# Function to add a message to chat history
def add_message(user_input, response):
    st.session_state.store['chat_history'].append({'user': user_input, 'bot': response})

# Streamlit UI
st.title("Chatbot Example")
user_input = st.text_input("Enter your message:")

if st.button("Send"):
    if user_input:
        # Simulate a bot response (replace with actual logic, e.g., calling Grok API)
        bot_response = f"Echo: {user_input}"
        add_message(user_input, bot_response)

# Display chat history
for chat in st.session_state.store['chat_history']:
    st.write(f"**You**: {chat['user']}")
    st.write(f"**Bot**: {chat['bot']}")
