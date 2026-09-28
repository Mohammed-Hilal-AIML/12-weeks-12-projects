from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "gpt2-dailydialog"

def main():
    if not (DATA / "train.jsonl").exists():
        raise FileNotFoundError("Run download_dataset.py and prepare_data.py first.")

    ds = load_dataset("json", data_files={
        "train": str(DATA / "train.jsonl"),
        "validation": str(DATA / "validation.jsonl")
    })

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(OUT),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tok["train"],
        eval_dataset=tok["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting GPT-2 fine-tuning")
    print("3 epochs | batch 4 | lr 5e-5 | fp16:", fp16)
    trainer.train()
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    print("Best/final model saved to:", OUT)

if __name__ == "__main__":
    main()
