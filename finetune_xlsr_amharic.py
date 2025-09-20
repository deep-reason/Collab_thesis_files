# ==========================
# XLS-R Wav2Vec2 Fine-Tuning (Colab/Kaggle-ready)
# ==========================
import os
import json
import re
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate

# ----------------------------
# Hard-coded defaults
# ----------------------------
DATASET_NAME = "AhunInteligence/w2v-bert-2.0-finetuning-amharic"
DATASET_CONFIG = "tr"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
PRETRAINED_MODEL = "facebook/wav2vec2-xls-r-300m"
REPO_NAME = "wav2vec2-large-xls-r-300m-tr-Collab"
OUTPUT_DIR = os.path.join("/kaggle/working", REPO_NAME)
NUM_TRAIN_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
EVAL_STEPS = 400
SAVE_STEPS = 400
LOGGING_STEPS = 100
FP16 = True
PUSH_TO_HUB = True
HUB_MODEL_ID = REPO_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True  # Optimize GPU throughput

# ----------------------------
# Preprocessing helpers
# ----------------------------
CHARS_TO_REMOVE = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(batch):
    if batch["sentence"] is None:
        batch["sentence"] = ""
    batch["sentence"] = re.sub(CHARS_TO_REMOVE, '', batch["sentence"]).lower()
    return batch

def replace_hatted_characters(batch):
    if batch["sentence"] is None:
        batch["sentence"] = ""
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    return batch

# ----------------------------
# Build vocabulary from dataset
# ----------------------------
def build_and_save_vocab(train_ds, test_ds, vocab_path="vocab.json"):
    def extract_all_chars(batch):
        all_text = " ".join(filter(None, batch["sentence"])) if isinstance(batch["sentence"], list) else batch["sentence"] or ""
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vt = train_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_ds.column_names)
    vs = test_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_ds.column_names)

    vocab_list = list(set(vt["vocab"][0]) | set(vs["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_path, 'w', encoding='utf-8') as vf:
        json.dump(vocab_dict, vf, ensure_ascii=False)

    print(f"Saved vocab ({len(vocab_dict)}) to {vocab_path}")
    return vocab_path

# ----------------------------
# Load and preprocess dataset
# ----------------------------
print("Loading dataset...")
raw_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
raw_test = load_dataset(DATASET_NAME, DATASET_CONFIG, split=EVAL_SPLIT)

print("Cleaning transcripts...")
raw_train = raw_train.map(remove_special_characters, num_proc=4)
raw_test = raw_test.map(remove_special_characters, num_proc=4)
raw_train = raw_train.map(replace_hatted_characters, num_proc=4)
raw_test = raw_test.map(replace_hatted_characters, num_proc=4)

vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
build_and_save_vocab(raw_train, raw_test, vocab_path=vocab_path)

tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token='[UNK]', pad_token='[PAD]', word_delimiter_token='|')
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

raw_train = raw_train.cast_column("audio", Audio(sampling_rate=16_000))
raw_test = raw_test.cast_column("audio", Audio(sampling_rate=16_000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

print("Processing audio & labels...")
train_dataset = raw_train.map(prepare_dataset, remove_columns=raw_train.column_names, batched=True, num_proc=4)
eval_dataset = raw_test.map(prepare_dataset, remove_columns=raw_test.column_names, batched=True, num_proc=4)

# ----------------------------
# Data collator
# ----------------------------
from typing import Any, Dict, List, Union

class DataCollatorCTCWithPadding:
    def __init__(self, processor: Wav2Vec2Processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ----------------------------
# Model
# ----------------------------
model = Wav2Vec2ForCTC.from_pretrained(
    PRETRAINED_MODEL,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_extractor()

# ----------------------------
# Metric
# ----------------------------
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ----------------------------
# TrainingArguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    evaluation_strategy="steps",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    gradient_checkpointing=True,
    fp16=FP16,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    save_total_limit=2,
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HUB_MODEL_ID if PUSH_TO_HUB else None,
    report_to=["tensorboard"],
    logging_dir=os.path.join(OUTPUT_DIR, "runs"),
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    ddp_find_unused_parameters=False,
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
)

# ----------------------------
# Train
# ----------------------------
print("Starting training...")
trainer.train()

# Save artifacts locally
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# Push to Hugging Face Hub
if PUSH_TO_HUB:
    try:
        print("Pushing final model to Hugging Face Hub...")
        trainer.push_to_hub(commit_message="Training completed", blocking=True)
        print("Push complete.")
    except Exception as e:
        print("Failed to push:", e)

print("All done. Outputs are in:", OUTPUT_DIR)
