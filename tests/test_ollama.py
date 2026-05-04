"""Test Ollama connection."""
import ollama

try:
    # List available models
    models = ollama.list()
    print("Available Ollama models:")
    for model in models.get('models', []):
        print(f"  - {model['name']}")

    if not models.get('models'):
        print("  No models found. Please install a model with: ollama pull llama3.2")

except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("\nMake sure Ollama is running!")
    print("Windows: Start the Ollama application")
    print("Or run: ollama serve")
