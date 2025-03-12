import os
from dotenv import load_dotenv
import openai
from langchain.prompts import PromptTemplate
from hugging_face_api_data import get_hf_api_response  # Import your HF API function

# Load environment variables and set the API key.
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_prompt(context):
    """
    Builds a prompt for the OpenAI model using the combined context.
    This context already includes the original prompt and the Hugging Face output.
    
    Args:
        context (str): A combined context that includes the original prompt and the
                       Hugging Face model's output.
    
    Returns:
        str: The generated response from the OpenAI model.
    """
    
    # Define the prompt template. Note that we now use just the context.
    template = """
    You are a helpful financial advisor chatbot. Use the provided information below to produce an enhanced response.
    
    {context}
    
    Answer:
    """
    
    # Create the PromptTemplate object.
    prompt = PromptTemplate(
        input_variables=["context"],
        template=template
    )

    # Format the prompt using the combined context.
    prompt_string = prompt.format(context=context)

    # Use the OpenAI Chat Completions API to generate the answer.
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor chatbot."},
                {"role": "user", "content": prompt_string}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "Error generating response."

if __name__ == "__main__":
    # Define your original prompt that you pass to the Hugging Face model.
    original_prompt = "What is the valuation of Indian Share market"
    
    # Optionally, get the Hugging Face API token from the environment.
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_hf_api_token_here")
    
    # Call the Hugging Face API function with the original prompt.
    hf_response = get_hf_api_response(
        prompt=original_prompt,
        max_new_tokens=250,
        temperature=0.8,
        repo_id="ai1-test/finance-chatbot-flan-t5-large",
        hf_api_token=hf_api_token
    )
    
    # Combine the original prompt and the Hugging Face output.
    # This single string will be used as the context in the OpenAI prompt.
    combined_context = (
        f"Original Prompt: {original_prompt}\n\n"
        f"Hugging Face Response:\n{hf_response}"
    )
    
    # Build the final prompt and get the enhanced response from OpenAI.
    final_response = build_prompt(combined_context)
    print("Final GPT Response:", final_response)
