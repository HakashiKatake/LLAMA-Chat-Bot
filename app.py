from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st

# Initialize Streamlit configuration
st.set_page_config(
    page_title="Llama2 Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Store chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# CSS Styling for the chat app
st.markdown("""
    <style>
    body { background-color: #1c1c1c; color: white; }
    .chat-container { max-width: 800px; margin: auto; }
    .user-message, .bot-message {
        margin: 10px 0; padding: 10px; border-radius: 8px;
    }
    .user-message { background-color: #4caf50; color: black; }
    .bot-message { background-color: #6c63ff; color: white; }
    .avatar { height: 40px; width: 40px; border-radius: 50%; display: inline-block; margin-right: 10px; }
    .user-avatar { background-color: #4caf50; }
    .bot-avatar { background-color: #673ab7; }
    .clear-button { background-color: #ff4d4d; border: none; color: white; border-radius: 8px; padding: 8px; cursor: pointer; }
    .export-button { background-color: #2196f3; color: white; padding: 8px 12px; border-radius: 8px; border: none; cursor: pointer; }
    .chat-input { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #673ab7; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for chat controls
with st.sidebar:
    st.title("🛠 Chat Controls")
    if st.button("Clear Chat History", help="Erase the current conversation."):
        st.session_state.chat_history = []

    # Export chat as a text file
    if st.session_state.chat_history:
        chat_content = "\n".join(
            [f"{chat['role'].capitalize()}: {chat['message']}" for chat in st.session_state.chat_history]
        )
        st.download_button(
            label="📥 Export Chat",
            data=chat_content,
            file_name="chat_history.txt",
            mime="text/plain",
        )

# Title
st.title("🦙 Llama2 Chatbot with LangChain")

# Initialize the chatbot prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond concisely and clearly."),
    ("human", "{question}")
])

# Initialize the Llama2 model
llm = OllamaLLM(model="llama2")

# Function to handle user input
def handle_input():
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        st.session_state.chat_history.append({"role": "user", "message": user_message})
        
        with st.spinner("Llama2 is thinking..."):
            full_prompt = prompt.format(question=user_message)
            response = llm.invoke(full_prompt)
        
        st.session_state.chat_history.append({"role": "bot", "message": response})
        st.session_state.user_input = ""  # Clear the input

# Chat input box
st.text_input(
    "💬 Ask a question:", 
    key="user_input",
    on_change=handle_input,
    placeholder="Type your message here...", 
    label_visibility="collapsed"
)

# Display chat history
with st.container():
    st.write("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(
                f"<div class='user-message'>"
                f"<div class='avatar user-avatar'></div><strong>You:</strong> {chat['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='bot-message'>"
                f"<div class='avatar bot-avatar'></div><strong>Llama2:</strong> {chat['message']}</div>",
                unsafe_allow_html=True,
            )
    st.write("</div>", unsafe_allow_html=True)
