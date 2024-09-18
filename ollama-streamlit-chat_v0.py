import streamlit as st
import requests
import json

def generate_ollama_response(prompt, model="tinyllama", stream=True):
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": stream
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers, stream=stream)
    
    if response.status_code == 200:
        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        full_response += content
                        yield content
                    if data.get('done', False):
                        break
        else:
            data = response.json()
            if 'message' in data and 'content' in data['message']:
                return data['message']['content']
            else:
                return "Error: Unexpected Response Format"
    else:
        return f"Error: {response.status_code}, {response.text}"

def main():
    st.title("Ollama Chat Interface")

    # Sidebar for model selection
    model = st.sidebar.selectbox("Choose a model", ["tinyllama", "llama2", "mistral"], index=0)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in generate_ollama_response(prompt, model=model):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
