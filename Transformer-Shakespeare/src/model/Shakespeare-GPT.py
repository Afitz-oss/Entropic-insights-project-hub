import torch
from pathlib import Path
import sys
import os

# Add training directory to path
sys.path.append(str(Path(__file__).parent.parent / 'trainning'))

from gpt2_shakespeare import GPTLanguageModel

def load_model(model_path=r'c:\Users\btofi\Downloads\Shakespearegpt_best (1).pt'):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with same parameters as training
    model = GPTLanguageModel(
        vocab_size=280,
        n_embd=128,
        block_size=64,
        n_head=8,
        n_layer=6,
        dropout=0.1
    ).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
        
    model.eval()
    return model

def generate_response(model, prompt, max_tokens=100, temperature=0.5):
    """Generate a response from the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Encode the prompt properly
    tokens = torch.tensor([min(ord(c), 279) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    generated_text = prompt  # Start with the prompt
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if tokens.size(1) > model.block_size:
                tokens = tokens[:, -model.block_size:]
            
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Convert token back to character, ensuring it's within valid range
            next_char = chr(min(next_token.item(), 279))
            generated_text += next_char
            
            # Fix: Ensure correct tensor dimensions for concatenation
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop if we generate a period followed by space or newline
            if len(generated_text) > 2 and generated_text[-2:] in ['. ', '.\n']:
                break
    
    return generated_text[len(prompt):]  # Return only the generated part

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def chat_interface():
    """Run the interactive chat interface."""
    print("Loading Shakespeare AI model...")
    model = load_model()
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
            continue
        
        if not user_input:
            continue
        
        # Generate response
        print("\nShakespeare: ", end="")
        response = generate_response(model, user_input)
        print(response)
        
        # Store in chat history
        chat_history.append(("You", user_input))
        chat_history.append(("Shakespeare", response))

if __name__ == "__main__":
    chat_interface()