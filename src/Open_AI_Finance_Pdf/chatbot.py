import os
import sys
from vector_store import init_faiss_vector_store
from retriever import retrieve_query
from prompt_template import get_finance_prompt
import embedder
import pdf_loader
import text_splitter
import nltk
from dotenv import load_dotenv
from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI()

load_dotenv()

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def create_chat_response(query, vector_store):
    """
    Generates a chat response using OpenAI based on the retrieved context.

    Args:
        query: The user's query.
        vector_store: The FAISS vector store.

    Returns:
        A string containing the chat response.
    """
    try:
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

            return response.choices[0].message
        else:
            return "Sorry, I couldn't find relevant information."

    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "An error occurred while processing your request."


if __name__ == "__main__":
    # Initialize Vector Store
    pdf_path = r"C:\End_to_End_Finance_ChatBot\data\ICICI-direct-FAQ.pdf"  # Replace with your PDF path
    text = pdf_loader.load_pdf_with_pypdf2(pdf_path)
    splitted_texts = text_splitter.split_text_with_nltk(text)
    vectors = embedder.generate_sentence_transformer_embeddings(splitted_texts)
    vector_store = init_faiss_vector_store(splitted_texts, vectors)

    if vector_store:
        while True:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                break
            response = create_chat_response(user_query, vector_store)
            print("Chatbot:", response)
    else:
        print("Failed to initialize vector store.")