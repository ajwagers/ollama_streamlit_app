import streamlit as st
import ollama
from ollama import Client

st.set_page_config(layout="wide")

# Function to get available models using Ollama client
def get_available_models(client):
    try:
        models = client.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return ["tinyllama", "phi3"]  # Fallback models
    
# Abstracted function for the API call with typing effect
def fetch_response(client, model, messages, message_placeholder):
    full_response = ""
    try:
        for chunk in client.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            full_response += chunk['message']['content']
            # Update the placeholder to simulate the AI typing response
            message_placeholder.markdown(full_response + "▌")
    except Exception as e:
        st.error(f"Error in chat: {e}")
    
    # Once the full response is complete, remove the "▌" cursor
    message_placeholder.markdown(full_response)
    return full_response

# Sidebar for host and model selection
with st.sidebar:
    st.title("Configuration")
    host = st.text_input("Ollama Host URL", value="http://localhost:11434")
    
    # Create Ollama client
    client = Client(host=host)
    
    st.title("Model Selection")
    models = get_available_models(client)
    
    # Find the index of "tinyllama:latest" in the list
    default_model = "tinyllama:latest"
    default_index = models.index(default_model) if default_model in models else 0
    
    # Set the selectbox default value to "tinyllama:latest"
    st.session_state.model = st.selectbox("Choose a model", models, index=default_index)

# Main chat interface
st.title("Streamlit Ollama Chat")

# This initializes an empty list for messages if it doesn't exist in the Streamlit session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# This loop displays all existing messages in the chat interface.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# This captures user input and adds it to the message history.
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # This section generates the AI response using the Ollama client directly, streams the response, and adds it to the message history.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = fetch_response(client, st.session_state.model, st.session_state.messages, message_placeholder)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display currently selected model and host in the sidebar
with st.sidebar:
    st.write(f"Currently using: {st.session_state.model}")
    st.write(f"Host: {host}")
