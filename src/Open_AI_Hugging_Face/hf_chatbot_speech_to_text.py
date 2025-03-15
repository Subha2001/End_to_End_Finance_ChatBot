import os
from dotenv import load_dotenv
import openai
from prompt_engineering import build_prompt
from hugging_face_api_data import get_hf_api_response
import tempfile

# Load environment variables from .env file.
load_dotenv()

# Set the OpenAI API key.
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_chat_response_with_tts(prompt):
    """
    Generates a chat response using Hugging Face and OpenAI, then converts it to speech.

    Args:
        prompt (str): The initial prompt for the Hugging Face model.

    Returns:
        tuple: (final_response, audio_path), where final_response is the text response,
               and audio_path is the path to the audio file (or None if speech generation fails).
    """
    try:
        # Get the response from the Hugging Face model.
        hf_response = get_hf_api_response(prompt)

        # Combine the original prompt and the Hugging Face output.
        combined_context = (
            f"Original Prompt: {prompt}\n\n"
            f"Hugging Face Response:\n{hf_response}"
        )

        # Use the build_prompt function to get the enhanced response from OpenAI.
        final_response = build_prompt(combined_context)

        # Text-to-speech
        try:
            speech = openai.audio.speech.create(
                input=final_response,
                speed=1.20,
                model="tts-1",
                voice="echo"
            )

            # Save the speech to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(speech.content)
                temp_audio_path = temp_audio.name

            return final_response, temp_audio_path

        except Exception as tts_e:
            print(f"Error generating speech: {tts_e}")
            return final_response, None

    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "Error generating response.", None

''''
# For Example
if __name__ == "__main__":
    prompt = "What is the valuation of Indian Share market"
    text_response, audio_path = generate_chat_response_with_tts(prompt)

    print("Final GPT Response:", text_response)

    if audio_path:
        print(f"Audio file saved to: {audio_path}")
        # Add code to play audio or provide download link here.
        # Example using playsound (install with: pip install playsound):
        # from playsound import playsound
        # playsound(audio_path)
    else:
        print("Audio generation failed.")

    # Interactive mode
    while True:
        user_prompt = input("Enter your prompt (or 'exit' to quit): ")
        if user_prompt.lower() == "exit":
            break
        text_response, audio_path = generate_chat_response_with_tts(user_prompt)
        print("Final GPT Response:", text_response)

        if audio_path:
            print(f"Audio file saved to: {audio_path}")
            # Add code to play audio or provide download link here.
            # Example using playsound (install with: pip install playsound):
            # from playsound import playsound
            # playsound(audio_path)
        else:
            print("Audio generation failed.")
            '''