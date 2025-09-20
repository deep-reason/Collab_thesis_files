# Install necessary libraries
# !pip install -q transformers datasets soundfile accelerate huggingface_hub resampy torchaudio

import os
import io
import time
import logging
import numpy as np
import torch
from functools import partial
from datasets import Dataset, Audio, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, HfFolder, get_full_repo_name, create_repo
import soundfile as sf
from torchaudio.transforms import Resample
import shutil

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Define the persistent log file path.
# '/kaggle/working/' is the only writable directory during a run,
# and its contents are saved to the "Output" tab after the run.
LOG_FILE_PATH = "/kaggle/working/processing_log.txt"

# Set up logging to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler() # Also prints to the console
    ]
)

# -----------------------------------------------------------------------------
# Configuration & Credentials
# -----------------------------------------------------------------------------

# You MUST set this environment variable in your Kaggle notebook secrets
# OR replace HfFolder.get_token() with your actual token string
HF_TOKEN = os.environ.get("HF_TOKEN", HfFolder.get_token())
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please set your Hugging Face token as a Kaggle Secret.")

# Define the target sampling rate
TARGET_SR = 16000

# -----------------------------------------------------------------------------
# Processing Functions
# -----------------------------------------------------------------------------

def is_normalized(sample, threshold=1e-6):
    """Checks if a dataset sample's audio is already peak-normalized."""
    try:
        audio_data = sample['audio']['array']
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if abs(np.mean(audio_data)) > threshold or not np.isclose(np.max(np.abs(audio_data)), 1.0, atol=threshold):
            return False
        return True
    except (KeyError, ValueError, TypeError):
        return False

def is_correct_sampling_rate(sample):
    """Checks if a dataset sample has the correct sampling rate."""
    try:
        return sample['audio']['sampling_rate'] == TARGET_SR
    except (KeyError, ValueError, TypeError):
        return False

def normalize_audio_sample(sample):
    """Applies normalization to a single audio sample."""
    audio_data = sample['audio']['array']
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data - np.mean(audio_data)
    peak_amp = np.max(np.abs(audio_data))
    if peak_amp > 1e-6:
        audio_data = audio_data / peak_amp
    sample['audio']['array'] = audio_data
    return sample

def resample_audio_sample(sample):
    """
    Applies resampling to a single audio sample, dynamically handling
    different original sampling rates.
    """
    audio = sample['audio']
    audio_data = audio['array']
    current_sr = audio['sampling_rate']

    if current_sr != TARGET_SR:
        # Dynamically create the Resample object for the current sample's SR
        resampler = Resample(orig_freq=current_sr, new_freq=TARGET_SR)
        
        # Convert numpy array to PyTorch tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Resample the tensor
        audio_data = resampler(audio_tensor).numpy()
    
    # Update the sample with the re-sampled audio data and new sampling rate
    sample['audio']['array'] = audio_data
    sample['audio']['sampling_rate'] = TARGET_SR
    return sample

# -----------------------------------------------------------------------------
# Main Orchestration Logic
# -----------------------------------------------------------------------------

def process_and_push_dataset(src_dataset_name, src_subset_name, dest_repo_name, num_workers=4, batch_size=500):
    """
    Optimized two-stage script that processes data and pushes to a new repository.
    
    Args:
        src_dataset_name (str): The name of the Hugging Face dataset to load.
        src_subset_name (str): The name of the dataset subset to load.
        dest_repo_name (str): The name of the new Hugging Face repository to push to.
        num_workers (int): Number of parallel processes to use for processing.
        batch_size (int): The number of samples to process in each batch before pushing.
    """
    api = HfApi()

    # Create the destination repository
    repo_url = get_full_repo_name(dest_repo_name)
    try:
        create_repo(repo_url, private=False, exist_ok=True, token=HF_TOKEN)
        logging.info(f"Successfully created or found destination repository '{repo_url}'.")
    except Exception as e:
        logging.error(f"Failed to create/find repository: {e}")
        return

    # Load the original dataset in streaming mode for memory efficiency
    try:
        ds = load_dataset(src_dataset_name, src_subset_name, split='train', streaming=True)
        # We need a non-streaming version to get the total count for the progress bar
        ds_non_stream = load_dataset(src_dataset_name, src_subset_name, split='train')
        total_samples = len(ds_non_stream)
    except Exception as e:
        logging.error(f"Failed to load dataset {src_dataset_name}: {e}")
        return
    
    logging.info(f"Loaded dataset with {total_samples} samples.")

    # Process and push in chunks
    logging.info("Starting processing and pushing to new repository...")
    start_time = time.time()
    batch_counter = 0
    total_processed = 0

    current_batch_list = []
    
    # Iterate through the streaming dataset
    for sample in ds:
        # Stage 1: Normalize only if needed
        if not is_normalized(sample):
            sample = normalize_audio_sample(sample)
        
        # Stage 2: Resample only if needed
        if not is_correct_sampling_rate(sample):
            sample = resample_audio_sample(sample)

        current_batch_list.append(sample)
        total_processed += 1

        if len(current_batch_list) >= batch_size:
            processed_batch = Dataset.from_list(current_batch_list)
            
            # Convert to DatasetDict to push to hub with a specified split
            processed_batch.push_to_hub(
                dest_repo_name,
                split='train',
                commit_message=f"Add processed batch {batch_counter}",
                private=False,
                token=HF_TOKEN,
                # Use append=True for adding to the new dataset
                append=True
            )
            logging.info(f"✅ Batch {batch_counter} pushed. Progress: {total_processed}/{total_samples} samples.")
            current_batch_list = []
            batch_counter += 1

    # Push any remaining samples in the last batch
    if current_batch_list:
        processed_batch = Dataset.from_list(current_batch_list)
        processed_batch.push_to_hub(
            dest_repo_name,
            split='train',
            commit_message=f"Add final processed batch {batch_counter}",
            private=False,
            token=HF_TOKEN,
            append=True
        )
        logging.info(f"✅ Final batch pushed. Progress: {total_processed}/{total_samples} samples.")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info("\n\n📊 **Final Statistics Summary**")
    logging.info("-----------------------------------")
    logging.info(f"Total time elapsed: {total_time:.2f} seconds")
    logging.info("Processing complete. The new dataset on the Hugging Face Hub is now normalized and at 16kHz.")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual dataset and repo names
    SRC_DATASET_NAME = "Helsinki-NLP/opus-100"
    SRC_SUBSET_NAME = "en-am"
    DEST_REPO_NAME = "your-username/my-fully-processed-amharic-dataset"

    process_and_push_dataset(
        src_dataset_name=SRC_DATASET_NAME,
        src_subset_name=SRC_SUBSET_NAME,
        dest_repo_name=DEST_REPO_NAME,
        num_workers=os.cpu_count(),
        batch_size=500
    )
