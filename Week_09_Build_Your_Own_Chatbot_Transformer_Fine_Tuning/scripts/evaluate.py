from pathlib import Path
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL = ROOT / "models" / "gpt2-dailydialog"

def main():
    if not (MODEL / "config.json").exists():
        raise FileNotFoundError("Run scripts/train.py first.")
    ds = load_dataset("json", data_files={"test": str(DATA / "test.jsonl")})["test"]
    tok = GPT2Tokenizer.from_pretrained(str(MODEL))
    model = GPT2LMHeadModel.from_pretrained(str(MODEL))
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    args = TrainingArguments(output_dir=str(ROOT/"outputs"/"evaluation"), per_device_eval_batch_size=4, report_to="none")
    trainer = Trainer(model=model, args=args, eval_dataset=ds, processing_class=tok, data_collator=collator)
    loss = trainer.evaluate()["eval_loss"]
    ppl = math.exp(loss) if loss < 20 else float("inf")
    print(f"Test loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
