from pathlib import Path
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "gpt2-dailydialog"

def main():
    if not (MODEL / "config.json").exists():
        raise FileNotFoundError("Run scripts/train.py first.")
    tok = GPT2Tokenizer.from_pretrained(str(MODEL))
    model = GPT2LMHeadModel.from_pretrained(str(MODEL))
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    history = []
    print("GPT-2 DailyDialog chatbot. Type quit/exit to stop.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user:
            continue
        history.append(f"SPEAKER_A: {user}")
        prompt = "\n".join(history) + "\nSPEAKER_B:"
        x = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            y = model.generate(
                **x, max_new_tokens=60, do_sample=True,
                temperature=0.7, top_p=0.9,
                no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id
            )
        full = tok.decode(y[0], skip_special_tokens=True)
        bot = full[len(prompt):].strip() if full.startswith(prompt) else full
        bot = bot.split("SPEAKER_A:")[0].split("SPEAKER_B:")[0].strip()
        bot = bot or "I am not sure how to respond."
        print("Bot:", bot)
        history.append(f"SPEAKER_B: {bot}")
        history = history[-12:]

if __name__ == "__main__":
    main()
