import requests
import json
import time

def generate_with_model(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def test_models():
    test_prompt = "Write a short poem about coding."
    models = ["llama3.2", "mistral"]
    
    print("Testing models with Ollama...")
    
    for model in models:
        print(f"\nTesting {model}...")
        print("-" * 40)
        
        start_time = time.time()
        response = generate_with_model(model, test_prompt)
        end_time = time.time()
        
        print(f"Response from {model}:")
        print(response)
        print(f"Generation time: {end_time - start_time:.2f} seconds")

def get_prompt():
    text = input("Enter the prompt: ")
    response = generate_with_model("llama3.2", text)
    print(response)

if __name__ == "__main__":
    get_prompt()