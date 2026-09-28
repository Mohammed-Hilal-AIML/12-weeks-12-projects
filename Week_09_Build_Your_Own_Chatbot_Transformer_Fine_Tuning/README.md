# Week 09 — Build Your Own Chatbot with Transformer Fine-Tuning

GPT-2 conversational chatbot fine-tuned on DailyDialog using Hugging Face Transformers.

## Workflow
DailyDialog → SPEAKER_A/SPEAKER_B formatting → GPT-2 BPE tokenization → 3-epoch fine-tuning → generation → CLI chat with history → perplexity + 10-prompt testing.

## Guide-aligned settings
- GPT-2
- 3 epochs
- batch size 4
- learning rate 5e-5
- fp16 automatically enabled when CUDA is available
- checkpoints every 500 steps
- early stopping based on evaluation loss
- temperature 0.7
- top_p 0.9
- held-out test perplexity

## Structure
```text
Week_09_Build_Your_Own_Chatbot_with_Transformer_Fine_Tuning/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── .gitkeep
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── generate.py
│   ├── chat.py
│   └── evaluate.py
├── models/.gitkeep
└── outputs/
    ├── README.md
    └── .gitkeep
```

## Windows / Python 3.12 — no venv

```powershell
py -3.12 --version
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
py -3.12 -c "import torch, transformers, datasets, accelerate; print('All dependencies OK')"
```

Check GPU:
```powershell
py -3.12 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Prepare DailyDialog
```powershell
py -3.12 scripts/download_dataset.py
py -3.12 scripts/prepare_data.py
```

Generated files:
```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

Each conversation uses:
```text
SPEAKER_A: ...
SPEAKER_B: ...
```

## Verify GPT-2
```powershell
py -3.12 scripts/generate.py --prompt "SPEAKER_A: Hello! How are you?`nSPEAKER_B:"
```

## Fine-tune
```powershell
py -3.12 scripts/train.py
```

The best model is saved under `models/gpt2-dailydialog/`.

## Test 10 prompts
```powershell
py -3.12 scripts/generate.py --test-10
```

## CLI chatbot
```powershell
py -3.12 scripts/chat.py
```
Type `quit` or `exit` to stop. Previous turns are retained as context.

## Perplexity
```powershell
py -3.12 scripts/evaluate.py
```

## Important
Do not claim a specific loss, perplexity, accuracy, response quality, or training time until you actually run the project. Results depend on hardware and the training run.

## GitHub
Repository name:
`Week_09_Build_Your_Own_Chatbot_with_Transformer_Fine_Tuning`

Description:
`GPT-2 chatbot fine-tuned on DailyDialog using Hugging Face Transformers, with generation controls, conversation history, and perplexity evaluation.`
