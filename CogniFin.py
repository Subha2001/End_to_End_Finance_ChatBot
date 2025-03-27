import streamlit as st
import sys
import os
import tempfile
from gtts import gTTS

# --- Ensure Python can find your modules by adding the paths to sys.path ---
current_dir = os.path.dirname(__file__)
chatbot_dir = os.path.join(current_dir, "src", "Open_AI_Finance_Pdf")
finbot_dir = os.path.join(current_dir, "src", "Open_AI_Hugging_Face")

if chatbot_dir not in sys.path:
    sys.path.append(chatbot_dir)
if finbot_dir not in sys.path:
    sys.path.append(finbot_dir)

# --- Import the modules ---
try:
    from src.Open_AI_Finance_Pdf import chatbot  # from src/Open_AI_Finance_Pdf/chatbot.py
    from src.Open_AI_Finance_Pdf import OpenAI_pdf_speech_to_text  # TTS module for OpenAI
except ImportError as e:
    st.error(f"Error importing chatbot module: {e}")

try:
    from src.Open_AI_Hugging_Face import hf_chatbot_text_to_speech  # from src/Open_AI_Hugging_Face/hf_chatbot_speech_to_text.py
except ImportError as e:
    st.error(f"Error importing Hugging Face speech_to_text module: {e}")

# --- Enhanced CSS for unique style and design ---
st.markdown(
    """
    <style>
    /* Import a modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap');

    /* Set a smooth gradient background across the entire page */
    body {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Container for the chat history with drop shadow */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 15px;
        background-color: #000000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Common style for each message, with fade-in animation */
    .message {
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px;
        line-height: 1.4;
        font-weight: 500;
        opacity: 0;
        animation: fadeIn 0.5s forwards;
    }
    
    /* Keyframes for fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* User messages styling: light cyan background, dark cyan text, and avatar */
    .user-message {
        background-color: #E0F7FA;
        text-align: right;
        color: #006064;
    }
    
    /* Bot messages styling: light yellow background, dark olive text, and avatar */
    .bot-message {
        background-color: #FFF9C4;
        text-align: left;
        color: #827717;
    }

    /* Adding custom avatar icons next to messages */
    .user-message::before {
        content: "🙂 ";
        font-size: 1.2em;
    }
    
    .bot-message::before {
        content: "🤖 ";
        font-size: 1.2em;
    }

    /* Header styling to create hierarchy */
    h1, h3, h4, h5, h6 {
        color: #1976D2;
        font-weight: 600;
    }

    /* Header styling of h2 */
    h2 {
        color: #eb0c0c;
        font-weight: 600;
    }

    /* Header styling of h2 */
    .st-emotion-cache-mka0c2 p {
       word-break: break-word;
       margin-bottom: 0px;
       font-size: 18px;
    }

    
    /* Custom scrollbar styling for the chat container */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background-color: #808080;
        border-radius: 4px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Fixed Title ---
st.title("Cognitive Finance Chatbot")

# --- Initialize conversation history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to render chat history using HTML styling.
def render_chat_history():
    chat_content = '<div class="chat-container">'
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_content += f'<div class="message user-message"><strong>You:</strong> {msg["content"]}</div>'
        else:
            chat_content += f'<div class="message bot-message"><strong>Bot:</strong> {msg["content"]}</div>'
    chat_content += "</div>"
    return chat_content

# Create a placeholder for the chat messages.
chat_placeholder = st.empty()
chat_placeholder.markdown(render_chat_history(), unsafe_allow_html=True)

# --- Create a placeholder for the audio output above the chat interface ---
audio_placeholder = st.empty()

# --- Sidebar Navigation ---
st.sidebar.title("Select Application")
app_choice = st.sidebar.selectbox("Choose an option", ["App FAQ", "Fin Bot"])

# --- Chat interface for App FAQ ---
if app_choice == "App FAQ":
    st.header("ICICI Direct FAQ Chatbot")
    # Use a form to neatly handle the user input.
    with st.form("faq_form", clear_on_submit=True):
        user_query = st.text_input("Enter your question for the ICICI Direct App FAQ:")
        submit_button = st.form_submit_button("Submit")
    
    if submit_button and user_query.strip():
        # Append user message to the conversation history.
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        # Display a spinner while processing.
        with st.spinner("Processing your request..."):
            try:
                response = chatbot.create_chat_response(user_query)
                # If response is a tuple, extract only the text part.
                response_text = response[0] if isinstance(response, tuple) else response
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Generate TTS audio using OpenAI's module.
                audio_file = OpenAI_pdf_speech_to_text.text_to_speech(response_text)
                if audio_file:
                    # Insert the audio player above the chat interface.
                    audio_placeholder.audio(audio_file)
                else:
                    st.error("Failed to generate audio using OpenAI TTS.")
            except Exception as e:
                st.error(f"Error getting response from App FAQ module: {e}")
        # Refresh the chat display.
        chat_placeholder.markdown(render_chat_history(), unsafe_allow_html=True)

# --- Chat interface for Fin Bot ---
elif app_choice == "Fin Bot":
    st.header("Fin Bot")
    with st.form("finbot_form", clear_on_submit=True):
        user_query = st.text_input("Enter your Financial Query:")
        submit_button = st.form_submit_button("Submit")
    
    if submit_button and user_query.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("Processing your request..."):
            try:
                response = hf_chatbot_text_to_speech.generate_chat_response_with_tts(user_query)
                if isinstance(response, tuple):
                    response_text = response[0]
                    audio_path = response[1]  # Retrieve the audio path.
                else:
                    response_text = response
                    audio_path = None
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                if audio_path:
                    # Insert the audio player above the chat interface.
                    audio_placeholder.audio(audio_path)
                else:
                    st.error("Failed to generate audio using HuggingFace and OpenAI TTS.")
            except Exception as e:
                st.error(f"Error getting response from Fin Bot module: {e}")
        chat_placeholder.markdown(render_chat_history(), unsafe_allow_html=True)