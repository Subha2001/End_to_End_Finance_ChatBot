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
    from src.Open_AI_Finance_Pdf import OpenAI_pdf_speech_to_text # Import the speech to text.
except ImportError as e:
    st.error(f"Error importing chatbot module: {e}")

try:
    from src.Open_AI_Hugging_Face import hf_chatbot_speech_to_text  # from src/Open_AI_Hugging_Face/hf_chatbot_speech_to_text.py
except ImportError as e:
    st.error(f"Error importing Hugging Face speech_to_text module: {e}")

# --- Initialize conversation history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# A function to render chat history as Markdown
def render_chat_history():
    chat_content = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_content += f"**User:** {msg['content']}  \n\n"
        else:
            chat_content += f"**Assistant:** {msg['content']}  \n\n"
    return chat_content

# Create a placeholder for the chat messages. This placeholder will be updated after each message.
chat_placeholder = st.empty()
chat_placeholder.markdown(render_chat_history())

# --- Begin the Streamlit App ---
st.sidebar.title("Select Application")
app_choice = st.sidebar.selectbox("Choose an option", ["App FAQ", "Fin Bot"])

# --- Fixed Title ---
st.title("Chatbot Application")

# --- Chat interface for App FAQ ---
if app_choice == "App FAQ":
    st.header("App FAQ Chatbot")
    user_query = st.text_input("Enter your question for App FAQ:")
    if st.button("Submit", key="faq_submit"):
        if user_query.strip():
            # Append user's query to the conversation history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            try:
                response = chatbot.create_chat_response(user_query)
                # If response is a tuple, extract only the text part.
                if isinstance(response, tuple):
                    response_text = response[0]
                else:
                    response_text = response
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                # Generate speech from the response using OpenAI's tts
                audio_file = OpenAI_pdf_speech_to_text.text_to_speech(response_text)

                if audio_file:
                    st.audio(audio_file)
                else:
                    st.error("Failed to generate audio using OpenAI TTS.")

            except Exception as e:
                st.error(f"Error getting response from App FAQ module: {e}")

            # Update the chat display
            chat_placeholder.markdown(render_chat_history())

# --- Chat interface for Fin Bot ---
elif app_choice == "Fin Bot":
    st.header("Fin Bot")
    user_query = st.text_input("Enter your query for Fin Bot:")
    if st.button("Submit", key="finbot_submit"):
        if user_query.strip():
            # Append user's query to the conversation history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            try:
                response = hf_chatbot_speech_to_text.generate_chat_response_with_tts(user_query)
                # If response is a tuple, extract only the text part.
                if isinstance(response, tuple):
                    response_text = response[0]
                    audio_path = response[1] # get the audio path from the response
                else:
                    response_text = response
                    audio_path = None

                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                # Play the audio if generated successfully
                if audio_path:
                    st.audio(audio_path)
                else:
                    st.error("Failed to generate audio using HuggingFace and OpenAI TTS.")

            except Exception as e:
                st.error(f"Error getting response from Fin Bot module: {e}")

            # Update the chat display
            chat_placeholder.markdown(render_chat_history())