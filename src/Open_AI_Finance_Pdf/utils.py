import os
from dotenv import load_dotenv

def load_environment_variables():
    """Loads environment variables from .env file."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return openai_api_key

if __name__ == "__main__":
    api_key = load_environment_variables()
    print(f"OpenAI API Key loaded: {api_key[:5]}...") # print first 5 characters of key.