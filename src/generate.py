
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.model import GPT, GPTConfig

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.9, top_k=50):
    """Load a trained GPT model and generate text from a given prompt"""
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, k=top_k)
            probs = torch.zeros_like(logits).scatter_(1, indices, values)
            probs = F.softmax(probs, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    
    output_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

    return output_text

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("data/kinyarwanda_bpe.json")
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load("results/model.pt", map_location="cpu"))

    text = generate_text(model, tokenizer, "Paul Kagame", max_new_tokens=100)
    print(text)
