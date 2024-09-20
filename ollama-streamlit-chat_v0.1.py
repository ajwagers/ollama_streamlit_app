import streamlit as st
import requests
import json

st.set_page_config(layout="wide")

# Define the base URL for the API
base_api_url = "http://localhost:11434/api"

# Function to get available models from the Ollama API
def get_available_models():
    try:
        response = requests.get(f"{base_api_url}/tags")
        if response.status_code == 200:
            models_data = response.json()  # Assuming the response is a list of model tags
            models = [model["name"] for model in models_data.get('models', [])]
            #print(models_data)
            return models
        else:
            st.error(f"Failed to fetch models: {response.status_code}")
            return ["tinyllama", "phi3"]  # Fallback models
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return ["tinyllama", "phi3"]  # Fallback models

# Sidebar for model selection
with st.sidebar:
    st.title("Model Selection")
    models = get_available_models()
    
    # Find the index of "tinyllama:latest" in the list
    default_model = "tinyllama:latest"
    default_index = models.index(default_model) if default_model in models else 0
    
    # Set the selectbox default value to "tinyllama:latest"
    st.session_state.model = st.selectbox("Choose a model", models, index=default_index)

# Main chat interface
st.title("Streamlit Ollama Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = f"{base_api_url}/chat"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        payload = {
            "model": st.session_state.model,
            "messages": st.session_state.messages,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json"
        }

        with requests.post(st.session_state.api_url, json=payload, headers=headers, stream=True) as response:
            for chunk in response.iter_lines():
                if chunk:
                    chunk_data = json.loads(chunk.decode('utf-8'))
                    full_response += chunk_data.get('message', {}).get('content', '')
                    message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Display currently selected model in the sidebar
with st.sidebar:
    st.write(f"Currently using: {st.session_state.model}")
