def get_finance_prompt(query: str) -> str:
    """
    Generates a prompt for finance-related queries.

    Args:
        query: The user's query.

    Returns:
        A formatted prompt string.
    """
    prompt = f"""
    You are a helpful financial assistant. Please answer the following question based on the provided context.

    Question: {query}

    Context:
    {{context}}

    Answer:
    """
    return prompt

'''
# For Example:
if __name__ == "__main__":
    # Example
    query1 = "What is ESOP?"
    prompt1 = get_finance_prompt(query1)
    print("Example 1: General Finance Prompt")
    print(prompt1)
    print("-" * 40)
    '''