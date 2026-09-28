from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)

def main():
    print("Downloading DailyDialog from Hugging Face...")
    ds = load_dataset("daily_dialog")
    print(ds)
    print("Download/cache complete.")

if __name__ == "__main__":
    main()
