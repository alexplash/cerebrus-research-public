

from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from encoder_models import (
    EEGConv2D,
    EEGFlattenedMLP,
    EEGGRUEncoder,
    EEGLSTMEncoder,
    EEGRNNEncoder,
    EEGSubtractiveConv2D,
    EEGSubtractiveTransformerEncoder,
    EEGTransformerEncoder,
    EEGInceptionEncoder,
    EEGInceptionNoPatch
)
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from collections import Counter
from typing import Any, Dict, List

DATA_PATH = "data/BCI-Competition-IV-2a-BNCI-2014-001"
ALL_FILES = [
    "A01E.mat",
    "A01T.mat",
    "A02E.mat",
    "A02T.mat",
    "A03E.mat",
    "A03T.mat",
    "A04E.mat",
    "A04T.mat",
    "A05E.mat",
    "A05T.mat",
    "A06E.mat",
    "A06T.mat",
    "A07E.mat",
    "A07T.mat",
    "A08E.mat",
    "A08T.mat",
    "A09E.mat",
    "A09T.mat",
]

NUM_EEG_CHANNELS = 22
WINDOW_SIZE = 875
SAMPLING_RATE = 250

BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-3
TEST_SIZE = 0.1
MODEL_FOLDER = "encoder_model_weights"

MODEL_ARCHITECTURE = EEGInceptionEncoder

SEEDS = [42, 7, 123, 2025]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class EEGWindowDataset(Dataset):
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        window = torch.tensor(sample['window'], dtype = torch.float32)
        label = torch.tensor(sample['label'], dtype = torch.long)
        
        return window, label


def gen_datasets(data_path, file_names, random_state):
    
    # Official paradigm:
    # t = 0s: fixation cross + warning tone
    # t = 2s: cue appears
    # t = 6s: motor imagery period ends
    #
    # We skip to 2.5s to avoid fixation and early reaction lag.
    MI_START_SEC = 2.5
    MI_END_SEC = 6.0
    
    MI_START_OFFSET = int(MI_START_SEC * SAMPLING_RATE)
    MI_END_OFFSET = int(MI_END_SEC * SAMPLING_RATE)
    
    full_samples = []
    
    for file_name in file_names:
        full_path = f"{data_path}/{file_name}"
        
        file_data = loadmat(full_path)
        file_data = file_data["data"]
        
        # Most files have 3 EOG calibration blocks first, then 6 EEG runs.
        # A04T has a shortened EOG block, so its EEG runs start earlier.
        if file_name == "A04T.mat":
            session_idx_range = range(1, 7)
        else:
            session_idx_range = range(3, 9)
        
        for session_idx in session_idx_range:
            data = file_data[0][session_idx][0][0]
            
            eeg_data = data[0]       # (timesteps, 25)
            trial_indices = data[1]  # (num_trials, 1)
            labels = data[2]         # (num_trials, 1)
            
            eeg_data = eeg_data[:, :NUM_EEG_CHANNELS].astype(np.float32)
            trial_indices = np.asarray(trial_indices).reshape(-1).astype(int)
            labels = np.asarray(labels).reshape(-1).astype(int)
            
            # MATLAB indexing -> Python indexing
            trial_indices = trial_indices - 1
            labels = labels - 1
            
            # Some MAT versions include artifact flags at data[5].
            # 0 = clean, 1 = artifact.
            artifacts = None
            try:
                artifacts = np.asarray(data[5]).reshape(-1).astype(int)
                if len(artifacts) != len(labels):
                    artifacts = None
            except Exception:
                artifacts = None
            
            for trial_idx in range(len(trial_indices)):
                label = labels[trial_idx]
                
                if label not in [0, 1, 2, 3]:
                    raise RuntimeError(
                        f"for {file_name}, session {session_idx + 1}, "
                        f"label found {label} that is not within [0, 1, 2, 3]"
                    )
                
                if artifacts is not None:
                    if artifacts[trial_idx] != 0:
                        continue
                
                trial_start = trial_indices[trial_idx]
                
                # Only use likely motor-imagery region.
                # For BCI IV 2a, trial t=2s to t=6s is the meaningful MI region.
                window_region_start = trial_start + MI_START_OFFSET
                window_region_end = trial_start + MI_END_OFFSET
                
                window = eeg_data[window_region_start:window_region_end]
                
                if window.shape != (WINDOW_SIZE, NUM_EEG_CHANNELS):
                    continue
                
                # Skip NaN/inf windows if present.
                if not np.isfinite(window).all():
                    continue
                
                full_samples.append({
                    "window": window.copy(), # (875, 22)
                    "label": int(label),
                    "file_name": file_name,
                    "session_index": int(session_idx),
                    "trial_index": int(trial_idx),
                    "window_start": int(window_region_start),
                    "window_end": int(window_region_end),
                    "mi_start_sec": MI_START_SEC,
                    "mi_end_sec": MI_END_SEC,
                })
        
    if len(full_samples) == 0:
        raise RuntimeError("No samples were generated. Check trial offsets/window size.")
    
    sample_labels = [s['label'] for s in full_samples]
    
    train_samples, test_samples = train_test_split(
        full_samples,
        test_size = TEST_SIZE,
        random_state=random_state,
        shuffle=True,
        stratify=sample_labels
    )

    print(f"train windows: {len(train_samples)}")
    print(f"test windows: {len(test_samples)}")
    print(f"train label counts: {Counter([s['label'] for s in train_samples])}")
    print(f"test label counts: {Counter([s['label'] for s in test_samples])}")
    
    return EEGWindowDataset(train_samples), EEGWindowDataset(test_samples)



def train_model(model_architecture, train_dataset):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = model_architecture().to(device)
    
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = LR)
    
    model_name = model_architecture.__name__
    print("\n\n\n")
    print(f"Training {model_name}")
    print(f"Device {device}")
    print("------------------------------------------------------------------------")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_samples = 0
        
        for windows, labels in train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(windows)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            predictions = torch.argmax(logits, dim = 1)
            
            epoch_train_loss += loss.item() * windows.size(0)
            epoch_train_correct += (predictions == labels).sum().item()
            epoch_train_samples += windows.size(0)
        
        epoch_loss = epoch_train_loss / epoch_train_samples
        epoch_accuracy = epoch_train_correct / epoch_train_samples
        
        print(f"Epoch {epoch + 1}: loss = {epoch_loss:.4f} | accuracy = {epoch_accuracy:.4f}")
    
    print("------------------------------------------------------------------------")
    
    os.makedirs(MODEL_FOLDER, exist_ok = True)
    
    return model


def evaluate_model(model_architecture, model, test_dataset):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)
    
    model.to(device)
    
    model.eval()
    
    model_name = model_architecture.__name__
    
    print(f"Testing {model_name}")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for windows, labels in test_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            
            logits = model(windows)
            
            preds = torch.argmax(logits, dim = 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total
    
    print(f"model accuracy = {acc}")
    
    print(f"validation completed for {model_name}.")
    print("------------------------------------------------------")
    
    return acc



def main():
    
    accuracy_table = {
        "model_name": [],
        "accuracy": [],
        "subject_id": [],
        "seed": [],
    }
    
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    for seed in SEEDS:
        print("\n")
        print(f"SEED {seed}")
        print("\n")
        
        for i in range(0, len(ALL_FILES), 2):
            
            file_1 = ALL_FILES[i+1]
            file_2 = ALL_FILES[i]
            
            files = [file_1, file_2]
            subject_id = int(file_1[1:3])
            
            set_seed(seed)
            
            train_dataset, test_dataset = gen_datasets(
                DATA_PATH,
                files,
                seed
            )

            print(f"train dataset: {len(train_dataset)}")
            print(f"test dataset: {len(test_dataset)}")
            
            
            set_seed(seed)
            
            model = train_model(MODEL_ARCHITECTURE, train_dataset)
            acc = evaluate_model(MODEL_ARCHITECTURE, model, test_dataset)
            
            accuracy_table["model_name"].append(MODEL_ARCHITECTURE.__name__)
            accuracy_table["accuracy"].append(acc)
            accuracy_table["subject_id"].append(subject_id)
            accuracy_table['seed'].append(seed)
                
            model_path = f"{MODEL_FOLDER}/{MODEL_ARCHITECTURE.__name__}_subject_{subject_id}_seed_{seed}.pth"
            torch.save(model.state_dict(), model_path)
    
    accuracy_df = pd.DataFrame(accuracy_table)
    
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    accuracy_df.to_csv(f"{MODEL_FOLDER}/accuracy.csv", index=False)


if __name__ == "__main__":
    main()
