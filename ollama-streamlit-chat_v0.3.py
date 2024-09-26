import streamlit as st
from ollama import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
import time

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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

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
    ollama_llm = ChatOllama(model=st.session_state.model)

    if st.button("Clear chat history", type="primary"):
        st.session_state["messages"] = []

def main():
    # Main chat interface
    st.title("Streamlit Ollama Chat")

    # This initializes an empty list for messages if it doesn't exist in the Streamlit session state.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # This loop displays all existing messages in the chat interface.
    for message in st.session_state.messages:
        st.chat_message(message.type).write(message.content)

    # This captures user input and adds it to the message history.
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        if not st.session_state.model:
            st.info("""Please select an existing Ollama model name to continue.
            Visit https://ollama.ai/library for a list of supported models.
            Restart the streamlit app after downloading a model using the `ollama pull <model_name>` command.
            It should become available in the list of available models.""")
            st.stop()

        # This section generates the AI response using the Ollama client directly, streams the response, and adds it to the message history.
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            # Show "Thinking..." animation
            with thinking_placeholder:
                for _ in range(3):
                    for dots in [".", "..", "..."]:
                        thinking_placeholder.markdown(f"Thinking{dots}")
                        time.sleep(0.3)

            stream_handler = StreamHandler(st.empty())
            llm = ChatOllama(model=st.session_state.model, streaming=True, callbacks=[stream_handler])
            
            thinking_placeholder.empty()

            response = llm(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))

# Run the Streamlit app
if __name__ == "__main__":
    main()