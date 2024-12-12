import torch

def text_generation(model, input_text, num_chars=1000, block_size=128):
    """
    Generate a sequence of text by predicting characters iteratively.
    
    args:
    ---
        model: The trained model.
        input_text (str): Initial input text.
        num_chars (int): Number of characters to generate.
        block_size (int): Maximum context size for the model (e.g., 128).
    
    returns:
    --------
        str: The full generated text.
    """

    model.model.eval()
    input_sentence = input_text
    final_sentence = input_text

    with torch.no_grad():
        for _ in range(num_chars):

            input_indices = [model.dataset.stoi[char] for char in input_sentence]
            x_test = torch.tensor(input_indices).unsqueeze(0)  
            
            output = model.model(x_test)
            
            proba = torch.softmax(output[0, -1, :], dim=-1)
            predicted_index = torch.multinomial(proba, 1).item()
            
            predicted_letter = [k for k, v_ in model.dataset.stoi.items() if v_ == predicted_index][0]
            
            final_sentence += predicted_letter
            input_sentence += predicted_letter
            
            if len(input_sentence) > block_size:
                input_sentence = input_sentence[-block_size:]

    return final_sentence
