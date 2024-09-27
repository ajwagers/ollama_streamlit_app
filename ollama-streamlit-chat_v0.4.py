import streamlit as st
from ollama import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
import time
import chromadb
import json

st.set_page_config(layout="wide")

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("saved_chats")

# Function to get available models using Ollama client
def get_available_models(client):
    try:
        models = client.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return ["tinyllama", "phi3"]  # Fallback models

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")
        time.sleep(0.02)

# Function to save chat
def save_chat(chat_name, messages):
    chat_data = json.dumps([{"type": m.type, "content": m.content} for m in messages])
    collection.upsert(
        ids=[chat_name],
        documents=[chat_data],
        metadatas=[{"name": chat_name}]
    )

# Function to load chat
def load_chat(chat_name):
    results = collection.get(ids=[chat_name])
    if results['documents']:
        chat_data = json.loads(results['documents'][0])
        return [HumanMessage(content=m['content']) if m['type'] == 'human' else AIMessage(content=m['content']) for m in chat_data]
    return []

# Function to get saved chat names
def get_saved_chats():
    results = collection.get()
    return [item['name'] for item in results['metadatas']] if results['metadatas'] else []

# Sidebar for host, model selection, and chat management
with st.sidebar:
    st.title("Configuration")
    host = st.text_input("Ollama Host URL", value="http://localhost:11434")
    
    client = Client(host=host)
    
    st.title("Model Selection")
    models = get_available_models(client)
    
    # Initialize the model in session state if it's not already there
    if "model" not in st.session_state:
        default_model = "tinyllama:latest"
        st.session_state.model = default_model if default_model in models else models[0]
    
    # Use the session state to maintain the selected model
    st.session_state.model = st.selectbox("Choose a model", models, index=models.index(st.session_state.model))

    st.title("Chat Management")
    chat_option = st.radio("Chat Options", ["New Chat", "Load Saved Chat"])

    if chat_option == "Load Saved Chat":
        saved_chats = get_saved_chats()
        if saved_chats:
            selected_chat = st.selectbox("Select a saved chat", saved_chats)
            if st.button("Load Selected Chat"):
                st.session_state.messages = load_chat(selected_chat)
                st.session_state.chat_name = selected_chat
        else:
            st.info("No saved chats available.")

    if st.button("Clear chat history", type="primary"):
        st.session_state.messages = []
        st.session_state.chat_name = ""

def main():
    st.title("Streamlit Ollama Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_name" not in st.session_state:
        st.session_state.chat_name = ""

    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.chat_name:
            st.session_state.chat_name = prompt[:30]  # Use first 30 chars of initial prompt as chat name

        if not st.session_state.model:
            st.info("""Please select an existing Ollama model name to continue.
            Visit https://ollama.ai/library for a list of supported models.
            Restart the streamlit app after downloading a model using the `ollama pull <model_name>` command.
            It should become available in the list of available models.""")
            st.stop()

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            with thinking_placeholder:
                for _ in range(3):
                    for dots in [".", "..", "..."]:
                        thinking_placeholder.markdown(f"Thinking{dots}")
                        time.sleep(0.3)
            
            thinking_placeholder.empty()

            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            
            llm = ChatOllama(
                model=st.session_state.model,
                base_url=host,
                streaming=True,
                callbacks=[stream_handler]
            )

            response = llm.invoke(st.session_state.messages)
            
            message_placeholder.markdown(stream_handler.text)
            
            st.session_state.messages.append(AIMessage(content=response.content))

    # Chat name input and save button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.chat_name = st.text_input("Chat Name", value=st.session_state.chat_name)
    with col2:
        if st.button("Save Chat"):
            save_chat(st.session_state.chat_name, st.session_state.messages)
            st.success(f"Chat '{st.session_state.chat_name}' saved successfully!")

with st.sidebar:
    st.write(f"Currently using: {st.session_state.model}")
    st.write(f"Host: {host}")

if __name__ == "__main__":
    main()
