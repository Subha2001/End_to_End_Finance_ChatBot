import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI()
load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def text_to_speech(text):
    """
    Converts a given text string to speech using OpenAI's audio.speech.create
    and saves it to a temporary MP3 file.

    Args:
        text: The text to convert.
        voice: The voice to use (defaults to "alloy").

    Returns:
        The file path to the generated MP3 file, or None if there was an error.
    """
    try:
        speech = client.audio.speech.create(
            input=text,
            speed=1.20,
            model="tts-1",
            voice="nova"
        )

        # Create a temporary file to store the speech
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(speech.content) #write the audio content to the file.
        temp_file.close() #close the file, so it can be accessed.

        return temp_file.name
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

'''
# For Example
if __name__ == "__main__":
    text = "Hello, this is a test of OpenAI text-to-speech."
    audio_file_path = text_to_speech(text)

    if audio_file_path:
        print(f"Audio file saved to: {audio_file_path}")
        # Add code here to play the audio file or provide a download link.
        # Example using playsound (install with: pip install playsound):
        # from playsound import playsound
        # playsound(audio_file_path)
    else:
        print("Text-to-speech conversion failed.")
        '''