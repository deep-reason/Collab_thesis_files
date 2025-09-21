# ==========================
# XLS-R Wav2Vec2 Fine-Tuning (Colab-ready, streaming-safe)
# ==========================
import os
import numpy as np
import torch
from datasets import load_dataset, Audio, DatasetDict
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
DATASET_CONFIG = "default"
TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
PRETRAINED_MODEL = "facebook/wav2vec2-xls-r-300m"
TOKENIZER_REPO = "AhunInteligence/w2v-bert-2.0-amharic-finetunining-tokenizer"
REPO_NAME = "wav2vec2-large-xls-r-300m-tr-Collab"
BASE_DIR = "/content"
OUTPUT_DIR = os.path.join(BASE_DIR, REPO_NAME)
NUM_TRAIN_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
EVAL_STEPS = 400
SAVE_STEPS = 400
LOGGING_STEPS = 50
FP16 = True
PUSH_TO_HUB = True
HUB_MODEL_ID = REPO_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True  # Optimize GPU throughput

# ----------------------------
# Load dataset (streaming safe)
# ----------------------------
print("Loading dataset...")
dataset = DatasetDict({
    "train": load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT),
    "valid": load_dataset(DATASET_NAME, DATASET_CONFIG, split=VALID_SPLIT),
    "test": load_dataset(DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT),
})

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# ----------------------------
# Processor
# ----------------------------
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(TOKENIZER_REPO)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# ----------------------------
# Safe preprocessing function
# ----------------------------
def prepare_dataset(batch):
    audio = batch["audio"]
    # Extract audio
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values
    # Tokenize text
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Processing datasets in streaming mode...")

train_dataset = dataset["train"].map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    batched=False,
    num_proc=2,
)

valid_dataset = dataset["valid"].map(
    prepare_dataset,
    remove_columns=dataset["valid"].column_names,
    batched=False,
    num_proc=2,
)

# Save preprocessed dataset to disk (avoid reprocessing every run)
disk_dataset_path = os.path.join(BASE_DIR, "amharic_w2v_prepared")
DatasetDict({
    "train": train_dataset,
    "valid": valid_dataset,
    "test": dataset["test"],  # keep test raw for later eval
}).save_to_disk(disk_dataset_path)

print(f"Saved preprocessed dataset to {disk_dataset_path}")

# ----------------------------
# Data collator
# ----------------------------
from typing import Dict, List, Union

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
    eval_dataset=valid_dataset,
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
