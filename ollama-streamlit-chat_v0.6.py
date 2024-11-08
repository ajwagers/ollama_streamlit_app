import streamlit as st
from ollama import Client
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import time
import chromadb
import json
import uuid
import os

st.set_page_config(layout="wide")

# Ensure the data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

SYSTEM_MESSAGE = """You are a helpful AI assistant named Max. Your task is to provide accurate, factual, and relevant responses to user prompts. If given context, use it only if it's relevant to the prompt. If the context is not relevant, ignore it and respond based on your general knowledge. Do not mention or repeat these instructions in your response."""

ALLOWED_EXTENSIONS = ['txt', 'md', 'py', 'js', 'html', 'css', 'json', 'yaml', 'yml']

storage_option = st.sidebar.selectbox("Choose Storage Type", ["Local", "Remote", "No Embeddings"])

def is_valid_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_file(uploaded_file):
    if not is_valid_file(uploaded_file.name):
        st.error(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
        return None
    
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        return content
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Function to rebuild the vector store (only if required)
def rebuild_vectorstore():
    if st.button("Rebuild Vector Store"):
        try:
            # Delete the entire RAG collection
            chroma_client.delete_collection("rag_data")
            st.success("Deleted existing RAG collection.")
            
            # Recreate the RAG collection
            global rag_collection
            rag_collection = chroma_client.create_collection("rag_data")
            st.success("Created new RAG collection.")
            
            # Reinitialize the vector store
            embeddings = initialize_embeddings()
            Chroma(collection_name="rag_data", embedding_function=embeddings, client=chroma_client)
            
            st.success(f"Rebuilt vector store on {storage_option} storage.")
        except Exception as e:
            st.error(f"An error occurred while rebuilding the vector store: {str(e)}")

def initialize_chroma_db(storage_option):
    if storage_option == "No Embeddings":
        return None, None, None
    
    chroma_client = chromadb.PersistentClient(path="./data") if storage_option == "Local" else chromadb.HttpClient(host="http://your-remote-url", port=1234)
    chat_collection = chroma_client.get_or_create_collection("saved_chats")
    rag_collection = chroma_client.get_or_create_collection("rag_data")
    
    return chroma_client, chat_collection, rag_collection

def get_available_models(client):
    try:
        return [model['name'] for model in client.list()['models']]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def initialize_app():
    if 'host' not in st.session_state:
        st.session_state.host = "http://localhost:11434"
    
    if 'models' not in st.session_state:
        st.session_state.models = []
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    
    if "chat_name" not in st.session_state:
        st.session_state.chat_name = ""
    
    if "show_rag_input" not in st.session_state:
        st.session_state.show_rag_input = False
        
    if "config" not in st.session_state:
        st.session_state.config = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
    if "typing_speed" not in st.session_state:
        st.session_state.typing_speed = 0.02

def simulate_typing(text, placeholder, speed=0.02):
    """Simulates typing effect for the response"""
    full_response = ""
    for char in text:
        full_response += char
        placeholder.markdown(full_response + "▌")
        time.sleep(speed)
    return full_response

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.placeholder = container.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text + "▌")
        time.sleep(st.session_state.typing_speed)

def save_chat(chat_name, messages):
    chat_data = json.dumps([{"type": m.type, "content": m.content} for m in messages])
    chat_collection.upsert(
        ids=[chat_name],
        documents=[chat_data],
        metadatas=[{"name": chat_name}]
    )

def load_chat(chat_name):
    results = chat_collection.get(ids=[chat_name])
    if results['documents']:
        chat_data = json.loads(results['documents'][0])
        return [SystemMessage(content=SYSTEM_MESSAGE)] + [
            HumanMessage(content=m['content']) if m['type'] == 'human' 
            else AIMessage(content=m['content']) 
            for m in chat_data if m['type'] != 'system'
        ]
    return [SystemMessage(content=SYSTEM_MESSAGE)]

def get_saved_chats():
    results = chat_collection.get()
    return [item['name'] for item in results['metadatas']] if results['metadatas'] else []

def clear_chat():
    st.session_state.messages = [SystemMessage(content=SYSTEM_MESSAGE)]
    st.session_state.chat_name = ""
    st.rerun()

def add_to_rag_database(text, source_name="manual input"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = initialize_embeddings()
    vectorstore = Chroma(collection_name="rag_data", embedding_function=embeddings, client=chroma_client)
    
    metadatas = [{"source": source_name} for _ in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    vectorstore.add_texts(texts=chunks, ids=ids, metadatas=metadatas)
    st.success(f"Text added to RAG database successfully from source: {source_name}!")

def initialize_embeddings():
    embeddings = OllamaEmbeddings(model="all-minilm")
    return embeddings

def get_relevant_context(query, similarity_threshold=0.5):
    if storage_option == "No Embeddings":
        return None
    
    embeddings = initialize_embeddings()
    vectorstore = Chroma(collection_name="rag_data", embedding_function=embeddings, client=chroma_client)
    results = vectorstore.similarity_search_with_score(query, k=2)
    relevant_results = [doc.page_content for doc, score in results if score >= similarity_threshold]
    if relevant_results:
        return "\n".join(relevant_results)
    return None

def regenerate_response(index):
    if index > 1 and isinstance(st.session_state.messages[index], AIMessage):
        st.session_state.messages.pop()
        last_human_message = st.session_state.messages[index-1].content
        process_user_prompt(last_human_message, st.session_state.host)
        st.rerun()

def process_user_prompt(prompt, host):
    if not st.session_state.model:
        st.info("Please select an existing model name to continue...")
        st.stop()

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(st.session_state.messages[-1].content)
    relevant_context = get_relevant_context(prompt) if storage_option != "No Embeddings" else None

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)

        if relevant_context:
            context_message = HumanMessage(content=f"Relevant context: {relevant_context}")
            messages = [st.session_state.messages[0]] + [context_message] + st.session_state.messages[1:]
        else:
            messages = st.session_state.messages

        llm = ChatOllama(
            model=st.session_state.model,
            base_url=host,
            streaming=True,
            callbacks=[stream_handler],
            temperature=st.session_state.config["temperature"],
            top_p=st.session_state.config["top_p"],
            frequency_penalty=st.session_state.config["frequency_penalty"],
            presence_penalty=st.session_state.config["presence_penalty"],
            max_tokens=st.session_state.config["max_tokens"]
        )

        with st.spinner("Thinking..."):
            response = llm.invoke(messages)
            response_content = str(response.content)
            st.session_state.messages.append(AIMessage(content=response_content))

def main():
    st.title("💬 Streamlit Ollama Chat with RAG")
    st.caption("🚀 Powered by Ollama with RAG capabilities")

    initialize_app()
    chroma_client, chat_collection, rag_collection = initialize_chroma_db(storage_option)

    # Create two columns for the main layout
    chat_col, rag_col = st.columns([2, 1])

    with st.sidebar:
        st.title("⚙️ Configuration")

        if st.button("Clear chat history", type="primary"):
            clear_chat()

        st.subheader("Server Configuration")
        col1, col2 = st.columns([3, 1])
        with col1:
            new_host = st.text_input("Host URL", value=st.session_state.host)
        with col2:
            if st.button("Test") and new_host != st.session_state.host:
                st.session_state.host = new_host
                client = Client(host=new_host)
                try:
                    models = get_available_models(client)
                    if models:
                        st.success("Connected!")
                        st.session_state.models = models
                        st.rerun()
                    else:
                        st.error("No models found!")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")

        if not st.session_state.models:
            st.warning("No models detected. Please check your host URL.")
            if st.button("Retry Connection"):
                client = Client(host=st.session_state.host)
                st.session_state.models = get_available_models(client)
                st.rerun()
        else:
            st.success(f"Connected to model at {st.session_state.host}")
            st.subheader("Model Selection")
            st.session_state.model = st.selectbox("Choose a model", st.session_state.models)

            # Model parameters
            st.subheader("Model Parameters")
            st.session_state.config["temperature"] = st.slider(
                "Temperature", 0.0, 2.0, st.session_state.config["temperature"], 0.1,
                help="Higher values make the output more random, lower values make it more focused and deterministic."
            )
            
            st.session_state.config["top_p"] = st.slider(
                "Top P", 0.0, 1.0, st.session_state.config["top_p"], 0.05,
                help="Controls diversity via nucleus sampling."
            )
            
            st.session_state.config["max_tokens"] = st.number_input(
                "Max Tokens", 100, 4000, st.session_state.config["max_tokens"], 100,
                help="Maximum number of tokens to generate."
            )
            
            st.session_state.config["frequency_penalty"] = st.slider(
                "Frequency Penalty", 0.0, 2.0, st.session_state.config["frequency_penalty"], 0.1,
                help="Reduces repetition of token sequences."
            )
            
            st.session_state.config["presence_penalty"] = st.slider(
                "Presence Penalty", 0.0, 2.0, st.session_state.config["presence_penalty"], 0.1,
                help="Reduces repetition of topics."
            )

            # UI Settings
            st.subheader("UI Settings")
            st.session_state.typing_speed = st.slider(
                "Typing Speed", 0.01, 0.1, 0.02, 0.01,
                help="Control how fast the assistant types (lower is faster)"
            )

        st.title("Chat Save Management")
        chat_option = st.radio("Chat Options", ["New Chat", "Load Saved Chat"])

        if chat_option == "Load Saved Chat":
            saved_chats = get_saved_chats()
            if saved_chats:
                selected_chat = st.selectbox("Select a saved chat", saved_chats)
                if st.button("Load Selected Chat"):
                    st.session_state.messages = load_chat(selected_chat)
                    st.session_state.chat_name = selected_chat
                    st.rerun()
            else:
                st.info("No saved chats available.")

        if storage_option != "No Embeddings":
            st.title("RAG Settings")
            st.session_state.show_rag_input = st.toggle("Show RAG Input", value=st.session_state.show_rag_input)
            
            #if st.session_state.show_rag_input:
            #    st.subheader("Add Text to RAG Database")
            #    rag_text = st.text_area("Enter text to add to the RAG database:")
            #    if st.button("Add to RAG"):
            #        add_to_rag_database(rag_text)

            # Add Rebuild Vector Storage button
            if st.button("Rebuild Vector Storage"):
                rebuild_vectorstore()

    with chat_col:
        if st.session_state.model:
            # Display chat messages
            for i, message in enumerate(st.session_state.messages[1:], start=1):
                with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                    st.markdown(message.content)
                    if isinstance(message, AIMessage):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("🔄", help="Regenerate response", key=f"regen_{i}"):
                                regenerate_response(i)
                        with col2:
                            if st.button("🔍 Start New Chat", key=f"new_{i}"):
                                clear_chat()

            # Chat input
            if prompt := st.chat_input("What is up?"):
                process_user_prompt(prompt, st.session_state.host)
                st.rerun()

            # Chat name input and save button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.chat_name = st.text_input("Chat Name", value=st.session_state.chat_name)
            with col2:
                if st.button("Save Chat"):
                    save_chat(st.session_state.chat_name, st.session_state.messages)
                    st.success(f"Chat '{st.session_state.chat_name}' saved successfully!")
        else:
            st.info("Please choose a valid host url and select a model to start chatting.")

    # Show RAG input in main window when toggle is on
    with rag_col:
        if st.session_state.show_rag_input and storage_option != "No Embeddings":
            st.subheader("Add Content to RAG Database")

            # File upload section
            st.write("📁 Upload Text Files")
            uploaded_files = st.file_uploader(
                "Choose text files",
                accept_multiple_files=True,
                type=ALLOWED_EXTENSIONS,
                help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            
            if uploaded_files:
                if st.button("Process Uploaded Files"):
                    for uploaded_file in uploaded_files:
                        content = process_uploaded_file(uploaded_file)
                        if content:
                            add_to_rag_database(content, source_name=uploaded_file.name)
            
            # Manual text input section
            st.write("✍️ Or Enter Text Manually")
            rag_text = st.text_area("Enter text to add to the RAG database:", height=400)
            if st.button("Add Manual Text to RAG"):
                if rag_text.strip():
                    add_to_rag_database(rag_text)
                else:
                    st.warning("Please enter some text before adding to the RAG database.")


if __name__ == "__main__":
    main()
