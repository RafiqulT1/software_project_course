import ollama
import sys

#REF: https://github.com/ollama/ollama-python

def check_ollama_status():
    """
    Check if Ollama is running and has models installed
    """
    try:
        response = ollama.list()
        models = response.get('models', [])
        if not models:
            print("No models found. Please install a model using 'ollama pull modelname'")
            print("Example: ollama pull llama2")
            sys.exit(1)
        return models
    except ConnectionError:
        print("Error: Cannot connect to Ollama. Please make sure it's running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def list_available_models():
    """
    List all available models in local Ollama instance
    """
    try:
        response = ollama.list()
        models = response.get('models', [])
        return [model.model for model in models]
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def generate_response(prompt, model="llama2"):
    """
    Generate a response using locally running Ollama instance
    """
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error: {str(e)}"

def chat_conversation(model="llama2"):
    """
    Start an interactive chat session with the model
    """
    print(f"Starting chat with {model} (type 'exit' to quit)")
    messages = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        try:
            response = ollama.chat(
                model=model,
                messages=[*messages, {'role': 'user', 'content': user_input}]
            )
            messages.append({'role': 'user', 'content': user_input})
            messages.append({'role': 'assistant', 'content': response['message']['content']})
            print(f"\nAssistant: {response['message']['content']}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Check Ollama status and get available models
    available_models = check_ollama_status()
    # Get the first available model
    default_model = available_models[0].model
    print(f"Available models: {[model.model for model in available_models]}")
    
    # Example of single response generation
    prompt = "Explain what is Python in one sentence."
    print("\nPrompt:", prompt)
    print("Response:", generate_response(prompt, model=default_model))