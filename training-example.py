# Used on Google Colab with T4 GPU

# !pip install datasets
from datasets import load_dataset
# !pip install transformers
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
# !pip install torch
import torch
from torch.utils.data import DataLoader
# !pip install librosa
import librosa
# !pip install numpy
import numpy as np
# !pip install os
import os
# !pip install shutil
import shutil
# !pip install evaluate
import evaluate
from evaluate import load
# !pip install tensorflow
import tensorflow as tf
# from google.colab import files  # Importation pour le téléchargement
os.environ["WANDB_DISABLED"] = "true"


if torch.cuda.is_available():
    print("GPU is available and will be used!")
    device = torch.device("cuda")
else:
    print("No GPU detected. Using CPU instead.")
    device = torch.device("cpu")

# Paths
MODEL_DIR = "./wav2vec_trained_model"

# Load dataset
dataset = load_dataset("lewtun/music_genres_small")

# Feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")

# Preprocessing function
TARGET_LENGTH = 16000 * 15  # 5 seconds at 16kHz


def preprocess_function(examples):
    audio_array = examples["audio"]["array"]
    sampling_rate = examples["audio"]["sampling_rate"]

    if sampling_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

    if len(audio_array) > TARGET_LENGTH:
        audio_array = audio_array[:TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(audio_array)
        audio_array = np.pad(audio_array, (0, padding), "constant")

    inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    return {"input_values": inputs["input_values"][0], "labels": examples["genre_id"]}

# Preprocess dataset
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"].map(preprocess_function, remove_columns=["audio", "song_id", "genre"])
test_dataset = train_test_split["test"].map(preprocess_function, remove_columns=["audio", "song_id", "genre"])

# Load or initialize model
if os.path.exists(MODEL_DIR):
    print(f"Loading model from {MODEL_DIR}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
    shutil.rmtree(MODEL_DIR)
else:
    print("No saved model found. Initializing a new model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=10, ignore_mismatched_sizes=True
    )

print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Collation function
def collate_fn(batch):
    input_values = torch.stack([torch.tensor(example["input_values"]) for example in batch])
    labels = torch.tensor([example["labels"] for example in batch])
    return {"input_values": input_values, "labels": labels}

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",
    fp16=True,
    gradient_accumulation_steps=2,
)

# Load accuracy metric
accuracy_metric = load("accuracy")
f1_metric = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_metric.compute(predictions=predictions.numpy(), references=labels)
    f1 = f1_metric.compute(predictions=predictions.numpy(), references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
# trainer.train()

# Save the model
trainer.save_model(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")
# Debug directory contents after saving
if os.path.exists(MODEL_DIR):
    print(f"Model directory contents after save: {os.listdir(MODEL_DIR)}")
else:
    print(f"Failed to save model to {MODEL_DIR}")

# Compress and download the model
# ARCHIVE_NAME = "wav2vec_trained_model.zip"
# print("Compressing the model directory...")
# shutil.make_archive("wav2vec_trained_model", 'zip', MODEL_DIR)
# print(f"Model compressed to {ARCHIVE_NAME}. Downloading...")
#
# files.download(ARCHIVE_NAME)  # Télécharge le fichier compressé

results = trainer.evaluate()
print("Evaluation results:", results)