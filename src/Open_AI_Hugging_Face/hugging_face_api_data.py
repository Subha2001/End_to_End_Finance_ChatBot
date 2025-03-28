import os
import requests

def get_hf_api_response(prompt: str,
                        max_new_tokens: int = 250,
                        temperature: float = 0.8,
                        repo_id: str = "ai1-test/finance-chatbot-flan-t5-large",
                        hf_api_token: str = os.getenv("HUGGINGFACE_API_KEY", "")) -> str:
    """
    Generates a response by sending the prompt directly to the Hugging Face Inference API.
    
    Parameters:
      prompt (str): The prompt to send to the model.
      repo_id (str): The identifier of the Hugging Face model.
      hf_api_token (str): Your Hugging Face API token.
      max_new_tokens (int): Maximum number of tokens to generate for the response.
      temperature (float): Sampling temperature; higher values yield more random outputs.
      
    Returns:
      str: The generated response from the model.
    """
    try:
        # Optionally set the API token in the environment
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token

        # Construct the API URL and headers
        api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
        headers = {"Authorization": f"Bearer {hf_api_token}"}

        # Build payload with generation parameters to request longer outputs
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
            }
        }
        
        # Execute the POST request
        response = requests.post(api_url, headers=headers, json=payload)

        # Check for a successful response
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")

        result = response.json()
        # Expected output is a list of dictionaries with a "generated_text" key
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            raise Exception("Unexpected response format: " + str(result))
    
    except Exception as e:
        print("Error while generating response:", e)
        return "Error generating response."


'''
# For Example
if __name__ == "__main__":
    
    # Define the prompt.
    prompt = "What is the valuation of Indian Share market"
    
    # Generate the response and print it.
    generated_response = get_hf_api_response(prompt)
    print("Generated Response:", generated_response)
    '''