import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'


def open_file(filepath):
    """Open a file and return its contents as a string."""
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while opening the file {filepath}: {e}")
        return ""


def get_relevant_context(rewritten_input, temp_embeddings, temp_content, top_k=3):
    """Get relevant context from the temp based on user input."""
    if temp_embeddings.nelement() == 0:
        return []

    input_embedding = ollama.embeddings(model='nomic-embed-text', prompt=rewritten_input)["embedding"]
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), temp_embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = [temp_content[idx].strip() for idx in top_indices]
    return relevant_context


def rewrite_query(user_input_json, conversation_history, ollama_model):
    """Rewrite the user query based on conversation history."""
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

    Return ONLY the rewritten query text, without any additional formatting or explanations.

    Conversation History:
    {context}

    Original query: [{user_input}]

    Rewritten query: 
    """
    try:
        response = client.chat.completions.create(
            model=ollama_model,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            n=1,
            temperature=0.1,
        )
        rewritten_query = response.choices[0].message.content.strip()
        return json.dumps({"Rewritten Query": rewritten_query})
    except Exception as e:
        print(f"Error during query rewriting: {e}")
        return json.dumps({"Rewritten Query": user_input})


def ollama_chat(user_input, system_message, temp_embeddings, temp_content, ollama_model, conversation_history):
    """Handle the chat interaction with context and rewriting."""
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {"Query": user_input}
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context(rewritten_query, temp_embeddings, temp_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    print("Debug1")

    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str

    print("Debug2")

    conversation_history[-1]["content"] = user_input_with_context

    messages = [{"role": "system", "content": system_message}] + conversation_history

    print("Debug3")

    try:
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages,
            max_tokens=500,
        )
        print("Debug4")
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during chat completion: {e}")
        return "An error occurred while processing your request."


def load_temp_content(filepath="temp.txt"):
    """Load the temp content from a file."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding='utf-8') as temp_file:
            return temp_file.readlines()
    else:
        print(f"temp file {filepath} does not exist.")
        return []


def generate_temp_embeddings(temp_content):
    """Generate embeddings for the temp content using Ollama."""
    temp_embeddings = []
    for content in temp_content:
        print(f"Content {content}")
        try:
            response = ollama.embeddings(model='nomic-embed-text', prompt=content)
            temp_embeddings.append(response["embedding"])
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            exit(1)
    return torch.tensor(temp_embeddings)


if __name__ == "__main__":
    print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
    parser = argparse.ArgumentParser(description="Ollama Chat")
    parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
    args = parser.parse_args()

    print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='llama3'
        )
    except Exception as e:
        print(f"Error initializing Ollama API client: {e}")
        exit(1)

    print(NEON_GREEN + "Loading temp content..." + RESET_COLOR)
    temp_content = load_temp_content()

    if temp_content:
        print(NEON_GREEN + "Generating embeddings for the temp content..." + RESET_COLOR)
        temp_embeddings_tensor = generate_temp_embeddings(temp_content)
        print("Embeddings for each line in the temp:")
        print(temp_embeddings_tensor)
    else:
        temp_embeddings_tensor = torch.tensor([])

    print("Starting conversation loop...")
    conversation_history = []
    system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

    while True:
        user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
        if user_input.lower() == 'quit':
            break

        response = ollama_chat(user_input, system_message, temp_embeddings_tensor, temp_content, args.model,
                               conversation_history)
        print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
