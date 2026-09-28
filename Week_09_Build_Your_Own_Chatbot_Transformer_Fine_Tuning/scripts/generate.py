from pathlib import Path
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "gpt2-dailydialog"

PROMPTS = [
"SPEAKER_A: Hello! How are you?\nSPEAKER_B:",
"SPEAKER_A: What do you like to do in your free time?\nSPEAKER_B:",
"SPEAKER_A: Can you help me with my problem?\nSPEAKER_B:",
"SPEAKER_A: What are your plans for today?\nSPEAKER_B:",
"SPEAKER_A: I am looking for a good restaurant.\nSPEAKER_B:",
"SPEAKER_A: What do you think about learning new skills?\nSPEAKER_B:",
"SPEAKER_A: I had a difficult day today.\nSPEAKER_B:",
"SPEAKER_A: Can you recommend something interesting to read?\nSPEAKER_B:",
"SPEAKER_A: What is your favorite kind of music?\nSPEAKER_B:",
"SPEAKER_A: I need advice about organizing my time.\nSPEAKER_B:"
]

def load():
    path = str(MODEL) if (MODEL / "config.json").exists() else "gpt2"
    tok = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tok, model, device

def reply(prompt, max_length=100):
    tok, model, device = load()
    x = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        y = model.generate(
            **x, max_length=max_length, do_sample=True,
            temperature=0.7, top_p=0.9,
            no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id
        )
    text = tok.decode(y[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt")
    p.add_argument("--test-10", action="store_true")
    a = p.parse_args()
    if a.test_10:
        for i, prompt in enumerate(PROMPTS, 1):
            print(f"\n--- Prompt {i} ---\n{prompt}\nResponse:\n{reply(prompt)}")
    else:
        print(reply(a.prompt or "SPEAKER_A: Hello, how are you?\nSPEAKER_B:"))

if __name__ == "__main__":
    main()
