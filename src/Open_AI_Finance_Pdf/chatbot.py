import os
from vector_store import init_faiss_vector_store
from retriever import retrieve_query
from prompt_template import get_finance_prompt
from dotenv import load_dotenv
from openai import OpenAI
from OpenAI_pdf_speech_to_text import text_to_speech

# Initialize OpenAI API client
client = OpenAI()

load_dotenv()

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def create_chat_response(query):
    """
    Generates a chat response using OpenAI based on the retrieved context,
    and converts the response to speech.
    
    Args:
        query: The user's query.
    
    Returns:
        A tuple (response_text, audio_file_path) where:
        - response_text: A string containing the chat response.
        - audio_file_path: The path to the MP3 file with the generated speech,
          or None if there was an error generating speech.
    """
    try:
        vector_store = init_faiss_vector_store()
        retrieved_docs = retrieve_query(vector_store, query, k=2)
    
        if retrieved_docs:
            context = "\n".join(retrieved_docs)
            prompt = get_finance_prompt(query).format(context=context)
            messages = [
                {"role": "assistant", "content": "You are a helpful Finance Assistant."},
                {"role": "user", "content": prompt}
            ]
    
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=250,
                temperature=0.7,
                n=1,
                stop=None,
            )
    
            chat_response = response.choices[0].message.content

            # Convert the chat response text to speech
            audio_file_path = text_to_speech(chat_response)
            return chat_response, audio_file_path
        else:
            return "Sorry, I couldn't find relevant information.", None
    
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "An error occurred while processing your request.", None

'''
# For Example
if __name__ == "__main__":
    user_query = "How do I open a trading account?"
    response_text, audio_file = create_chat_response(user_query)
    print("Response text:", response_text)
    if audio_file:
        print("Audio file generated at:", audio_file)'
        '''