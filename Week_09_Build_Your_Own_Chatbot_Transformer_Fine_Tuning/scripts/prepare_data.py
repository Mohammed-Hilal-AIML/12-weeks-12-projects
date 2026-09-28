from pathlib import Path
import json
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def format_dialog(dialog):
    return "\n".join(
        f"{'SPEAKER_A' if i % 2 == 0 else 'SPEAKER_B'}: {text}"
        for i, text in enumerate(dialog)
    )

def main():
    ds = load_dataset("daily_dialog")
    for split in ("train", "validation", "test"):
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(json.dumps({"text": format_dialog(row["dialog"])}, ensure_ascii=False) + "\n")
        print(f"{split}: {path}")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
