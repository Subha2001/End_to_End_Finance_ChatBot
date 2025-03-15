import openai
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def build_prompt(context: str) -> str:
    """
    Builds a prompt for the OpenAI model using the combined context.
    This context already includes the original prompt and the Hugging Face output.

    Args:
        context (str): A combined context that includes the original prompt 
                        and the Hugging Face model's output.

    Returns:
        str: The generated response from the OpenAI model.
    """
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

    try:
        # Use the OpenAI Chat Completion API to generate the answer.
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

''''
# For Example
if __name__ == "__main__":
    combined_context = """
    Original Prompt: What are the benefits of investing in mutual funds?
    Hugging Face Output: Mutual funds offer diversification and professional management.
    """
    response = build_prompt(combined_context)
    print(response)
'''
if __name__ == "__main__":
    combined_context = """
    Original Prompt: What are the benefits of investing in mutual funds?
    Hugging Face Output: Mutual funds offer diversification and professional management.
    """
    response = build_prompt(combined_context)
    print(response)