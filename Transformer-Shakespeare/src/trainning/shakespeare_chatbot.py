import torch
from pathlib import Path
import sys
import os
import tiktoken
from contextlib import nullcontext

# Add training directory to path
sys.path.append(str(Path(__file__).parent.parent / 'trainning'))

# Import model classes directly from Train_shakespeare2
from Train_shakespeare2 import GPT, Config

def load_model(model_path=r'c:\Users\btofi\Downloads\ckpt (1).pt'):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get config from checkpoint
        config_dict = checkpoint['config']
        config = Config(**config_dict)
        
        # Initialize model with loaded config
        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        print("Model loaded successfully!")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    model.eval()
    return model

def generate_response(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate a response from the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Add conversational context to the prompt
    context_prompt = f"[A friendly conversation in Shakespeare's style]\nPerson: {prompt}\nShakespeare: "
    
    # Encode with tiktoken
    enc = tiktoken.get_encoding("gpt2")
    context = torch.tensor(enc.encode(context_prompt), dtype=torch.long, device=device)
    context = context.unsqueeze(0)  # Add batch dimension
    
    # Setup autocast - only use CUDA if available
    if torch.cuda.is_available():
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    else:
        ctx = nullcontext()
    
    with torch.no_grad():
        with ctx:
            output = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            response = enc.decode(output[0].tolist())
    
    # Clean up the response
    response = response[len(context_prompt):].strip()
    # Remove character names and stage directions
    response = '\n'.join(line for line in response.split('\n') 
                        if not line.strip().isupper() and 
                        not line.startswith('[') and 
                        not line.startswith('Enter') and 
                        not line.startswith('Exit'))
    return response.strip()

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def chat_interface():
    """Run the interactive chat interface."""
    # Get the model path from environment variable or use default
    model_path = os.getenv('SHAKESPEARE_MODEL_PATH', r'c:\Users\btofi\Downloads\ckpt (1).pt')
    
    print(f"Loading Shakespeare AI model from {model_path}...")
    model = load_model(model_path)
    clear_screen()
    
    print("🎭 Welcome to Shakespeare Chat! 🎭")
    print("I shall speak in the manner of the Bard himself.")
    print("(Type 'quit' to exit, 'clear' to clear the chat)")
    print("-" * 50)
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nFarewell, dear friend! 👋")
            break
        
        if user_input.lower() == 'clear':
            clear_screen()
            chat_history = []
            print("🎭 Welcome to Shakespeare Chat! 🎭")
            print("(Type 'quit' to exit, 'clear' to clear the chat)")
            print("-" * 50)
            continue
        
        if not user_input:
            continue
        
        try:
            # Generate response
            print("\nShakespeare: ", end="", flush=True)
            response = generate_response(model, user_input)
            print(response)
            
            # Store in chat history
            chat_history.append(("You", user_input))
            chat_history.append(("Shakespeare", response))
        except Exception as e:
            print(f"\nApologies, dear friend, but I seem to have lost my words: {str(e)}")

if __name__ == "__main__":
    chat_interface()