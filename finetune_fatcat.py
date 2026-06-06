"""
Stable GPT-OSS-20B Fine-tuning (Safer Version)
- Proper chat template
- No manual role flattening
- No packing
- Safer LoRA config
- Safer LR
- Dataset validation
"""

import os
import json
import torch

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# =========================================================
# Config
# =========================================================
DATA_PATH  = "/mnt/raid0/home/nukliliang/fatcat_gpt/fatcat_merged.jsonl"

OUTPUT_DIR = "/mnt/raid0/home/nukliliang/fatcat_gpt/output_optdata"

SAVE_PATH  = "/mnt/raid0/home/nukliliang/fatcat_gpt/fatcat-lora"


MAX_SEQ_LEN = 1024

LORA_RANK = 256
LORA_DROPOUT = 0.05

EPOCHS = 3

BATCH_SIZE = 4
GRAD_ACCUM = 8

LR = 5e-5


# =========================================================
# Stability
# =========================================================
os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "0"
os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================================================
# Load model
# =========================================================
print("Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=MAX_SEQ_LEN,

    load_in_4bit=True,
    dtype=None,

    full_finetuning=False,
)


# =========================================================
# Tokenizer fixes
# =========================================================
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("\n========== TOKENIZER INFO ==========")
print("EOS:", tokenizer.eos_token)
print("PAD:", tokenizer.pad_token)
print("PAD ID:", tokenizer.pad_token_id)
print("CHAT TEMPLATE EXISTS:", tokenizer.chat_template is not None)
print("====================================\n")


# =========================================================
# Dataset loader
# =========================================================
def load_jsonl_dataset(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                if "messages" not in item:
                    print(f"[Skip] line {i}: no messages")
                    continue

                if not isinstance(item["messages"], list):
                    print(f"[Skip] line {i}: messages not list")
                    continue

                valid = True

                for msg in item["messages"]:

                    if "role" not in msg:
                        valid = False
                        break

                    if "content" not in msg:
                        valid = False
                        break

                    if not isinstance(msg["content"], str):
                        valid = False
                        break

                    if len(msg["content"].strip()) == 0:
                        valid = False
                        break

                if not valid:
                    print(f"[Skip] invalid sample at line {i}")
                    continue

                data.append(item)

            except Exception as e:
                print(f"[Skip] line {i}: {e}")

    return data


print("Loading dataset...")
raw_data = load_jsonl_dataset(DATA_PATH)

print(f"Valid samples: {len(raw_data)}")

if len(raw_data) == 0:
    raise ValueError("Dataset is empty!")


# =========================================================
# Format dataset using chat template
# =========================================================
def format_chat(example):

    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    text += tokenizer.eos_token

    return {"text": text}


formatted_data = []

for sample in raw_data:

    try:
        formatted = format_chat(sample)

        if len(formatted["text"].strip()) == 0:
            continue

        formatted_data.append(formatted)

    except Exception as e:
        print("Format error:", e)


dataset = Dataset.from_list(formatted_data)

print("\n========== SAMPLE ==========\n")
print(dataset[0]["text"][:2000])
print("\n============================\n")


# =========================================================
# Split
# =========================================================
split = dataset.train_test_split(
    test_size=0.05,
    seed=42,
)

train_dataset = split["train"]
eval_dataset  = split["test"]

print("Train samples:", len(train_dataset))
print("Eval samples :", len(eval_dataset))


# =========================================================
# LoRA
# =========================================================
print("\nApplying LoRA...")

model = FastLanguageModel.get_peft_model(
    model,

    r=LORA_RANK,

    lora_alpha=LORA_RANK * 2,

    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",

    use_gradient_checkpointing="unsloth",

    use_rslora=False,
)

model.print_trainable_parameters()


# =========================================================
# Trainer
# =========================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    resume_from_checkpoint=False,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    dataset_text_field="text",

    max_seq_length=MAX_SEQ_LEN,

    packing=False,

    args=TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LR,

        lr_scheduler_type="cosine",

        warmup_steps=30,

        logging_steps=5,
        eval_steps=30,
        save_steps=50,

        save_total_limit=10,

        per_device_eval_batch_size=1,

        max_grad_norm=1.0,

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        optim="adamw_8bit",

        gradient_checkpointing=True,

        report_to="none",

        remove_unused_columns=False,
    ),
)


# =========================================================
# Train
# =========================================================
print("\nStarting training...\n")

trainer.train()


# =========================================================
# Save
# =========================================================
print("\nSaving model...")

model.save_pretrained(SAVE_PATH)

tokenizer.save_pretrained(SAVE_PATH)

print("\nDone!")
print("Saved to:", SAVE_PATH)