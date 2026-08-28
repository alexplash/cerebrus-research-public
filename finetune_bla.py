

from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from encoder_models import (
    EEGInceptionEncoder,
)
from bla_models import BLA
from simulation_validation import SimulationValidator
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from collections import Counter
from typing import Any, Dict, List
from itertools import permutations
import ray.train
from ray.train import ScalingConfig
from ray.train.torch import (
    TorchTrainer,
    prepare_model,
    prepare_data_loader,
    get_device,
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SUBJECTS = [1, 3, 7, 8]
WEIGHTS_ROOT = PROJECT_ROOT / "encoder_model_weights"
DATA_ROOT = PROJECT_ROOT / "data" / "BCI-Competition-IV-2a-BNCI-2014-001"
WEIGHT_PATHS = [
    WEIGHTS_ROOT / f"EEGInceptionEncoder_subject_{i}_seed_2025.pth"
    for i in SUBJECTS
]
DATA_PATH = DATA_ROOT
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

LABEL_TO_EEG_COMMAND = {
    0: "left-hand imagery",
    1: "right-hand imagery",
    2: "feet imagery",
    3: "tongue imagery",
}

FLIGHT_COMMAND_TO_TOKENS = {
    "fly forward":  ["<ACT_2>", "<ACT_1>", "<ACT_1>"],
    "fly backward": ["<ACT_0>", "<ACT_1>", "<ACT_1>"],
    "turn left":    ["<ACT_1>", "<ACT_2>", "<ACT_1>"],
    "turn right":   ["<ACT_1>", "<ACT_0>", "<ACT_1>"],
    "ascend":       ["<ACT_1>", "<ACT_1>", "<ACT_2>"],
    "descend":      ["<ACT_1>", "<ACT_1>", "<ACT_0>"],
    "hover":        ["<ACT_1>", "<ACT_1>", "<ACT_1>"],
}


NUM_EEG_CHANNELS = 22
WINDOW_SIZE = 875
SAMPLING_RATE = 250

NUM_GPUS = 16

RUN_SIM_VAL = False
SIM_MANIFEST_PATH = "sim_manifest/sim_manifest.json"

TEST_SIZE = 0.1

# specify your metrics output path here
METRICS_PATH = None

BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-3

SEEDS = [2025]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
class EEGActionDataset(Dataset):
    
    def __init__(self, samples, tokenizer, action_dim: int = 3):
        self.samples = samples
        self.tokenizer = tokenizer
        self.action_dim = action_dim
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        eeg_window = torch.tensor(sample['window'], dtype = torch.float32)
        
        instruction = sample['instructions']
        action_tokens = sample['action_tokens']
        
        if len(action_tokens) != self.action_dim:
            raise ValueError(
                f"expected {self.action_dim} action tokens. "
                f"received {len(action_tokens)}."
            )
        
        action_token_ids = self.tokenizer.convert_tokens_to_ids(
            action_tokens
        )
        
        if any(token_id is None for token_id in action_token_ids):
            raise ValueError(f"could not tokenize action sequence: {action_tokens}")
        
        if self.tokenizer.unk_token_id is not None:
            if self.tokenizer.unk_token_id in action_token_ids:
                raise ValueError(...)
        
        action_token_ids = torch.tensor(
            action_token_ids,
            dtype=torch.long
        )
        
        return {
            "eeg": eeg_window,
            "instruction": instruction,
            "target_action_token_ids": action_token_ids,
        }
        
        
def gen_instruction_variants():

    instruction_variations = []
    flight_control_mappings = []

    eeg_commands = list(LABEL_TO_EEG_COMMAND.values())
    flight_commands = list(FLIGHT_COMMAND_TO_TOKENS.keys())

    for command_permutation in permutations(
        flight_commands,
        len(eeg_commands),
    ):
        flight_control_mapping = dict(
            zip(eeg_commands, command_permutation)
        )

        instruction = (
            "Control the drone using the following "
            "mental-command mapping:\n\n"
            f"- Left-hand imagery: "
            f"{flight_control_mapping['left-hand imagery']}\n"
            f"- Right-hand imagery: "
            f"{flight_control_mapping['right-hand imagery']}\n"
            f"- Feet imagery: "
            f"{flight_control_mapping['feet imagery']}\n"
            f"- Tongue imagery: "
            f"{flight_control_mapping['tongue imagery']}\n\n"
            "Execute the action associated with the current "
            "brain signal."
        )

        instruction_variations.append(instruction)
        flight_control_mappings.append(
            flight_control_mapping
        )

    expected_variants = 840

    if len(instruction_variations) != expected_variants:
        raise RuntimeError(
            "Expected 840 instruction variants, but generated "
            f"{len(instruction_variations)}."
        )

    return instruction_variations, flight_control_mappings
    
    


def gen_datasets(
    subject, 
    data_path, 
    file_names, 
    instruction_variations, 
    flight_control_mappings, 
    tokenizer,
    random_state
):
    
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
    
    subject_id = f"{subject:02d}"

    subject_file_names = [
        f"A{subject_id}T.mat",
        f"A{subject_id}E.mat",
    ]
    
    full_samples = []
    
    for file_name in subject_file_names:
        full_path = Path(data_path) / file_name
        
        file_data = loadmat(str(full_path))
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
                
                eeg_window = window.copy()
                
                for i in range(len(instruction_variations)):
                    instruction_variant = instruction_variations[i]
                    flight_control_mapping = flight_control_mappings[i]
                    
                    eeg_command = LABEL_TO_EEG_COMMAND[int(label)]
                    flight_command = flight_control_mapping[eeg_command]
                    
                    action_tokens = FLIGHT_COMMAND_TO_TOKENS[flight_command]
                
                    full_samples.append({
                        "window": eeg_window, # (875, 22)
                        "label": int(label),
                        "instructions": instruction_variant,
                        "action_tokens": action_tokens,
                    })
    if len(full_samples) == 0:
        raise RuntimeError("No samples were generated. Check trial offsets/window size.")
    
    sample_labels = [s['label'] for s in full_samples]
    
    train_samples, test_samples = train_test_split(
        full_samples,
        test_size=TEST_SIZE,
        random_state=random_state,
        shuffle=True,
        stratify=sample_labels
    )  
    
    print(f"train windows: {len(train_samples)}")
    print(f"test windows: {len(test_samples)}")
    print(f"train label counts: {Counter([s['label'] for s in train_samples])}")
    print(f"test label counts: {Counter([s['label'] for s in test_samples])}")
        

    return EEGActionDataset(train_samples, tokenizer), EEGActionDataset(test_samples, tokenizer)


def gen_train_func_for_ray_subject_and_seed(
    seed, 
    subject, 
    weights_path,
    EEGInceptionEncoder,
    BLA,
    gen_datasets,
    DATA_PATH,
    ALL_FILES,
    instruction_variations,
    flight_control_mappings
):
    
    def train_and_val_loop_per_worker(config):
        
        set_seed(seed)
        
        device = get_device()

        brain_encoder = EEGInceptionEncoder()
        brain_encoder_state_dict = torch.load(str(weights_path), map_location="cpu")
        brain_encoder.load_state_dict(brain_encoder_state_dict)

        base_model = BLA(
            brain_encoder=brain_encoder,
            device=device,
        )

        tokenizer = base_model.get_tokenizer()
        model = prepare_model(base_model)
        
        train_dataset, test_dataset = gen_datasets(
            subject,
            DATA_PATH,
            ALL_FILES,
            instruction_variations,
            flight_control_mappings,
            tokenizer,
            seed
        )
        
        print("expanded train samples:", len(train_dataset))
        print("expanded test samples:", len(test_dataset))
        
        set_seed(seed)
        
        # ----------------------------------------------------
        # training loop start
        # ----------------------------------------------------
        train_loader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True)
        train_loader = prepare_data_loader(train_loader)
        
        world_size = ray.train.get_context().get_world_size()
        world_rank = ray.train.get_context().get_world_rank()
        
        optimizer = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config['lr']
        )
        
        for epoch in range(config['epochs']):
            model.train()
            
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            
            epoch_train_loss = 0
            epoch_train_correct = 0
            epoch_train_samples = 0
            batch_count = 0
            
            for batch in train_loader:
                batch_count += 1
                eeg = batch['eeg']
                instructions = batch['instruction']
                target_action_token_ids = batch["target_action_token_ids"]
                
                optimizer.zero_grad()
                
                outputs = model(
                    eeg=eeg,
                    instructions=instructions,
                    target_action_token_ids=target_action_token_ids
                )
                
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                
                prediction_ids = outputs["predicted_action_token_ids"]
                target_action_token_ids = target_action_token_ids.to(device)
                matches = prediction_ids == target_action_token_ids # (B, 3) of boolean values
                
                epoch_train_loss += loss.item()
                epoch_train_samples += (len(instructions) * 3)
                epoch_train_correct += matches.sum().item()
            
            stats = torch.tensor(
                [epoch_train_loss, epoch_train_correct, epoch_train_samples, batch_count],
                device=device,
                dtype=torch.float64,
            )

            epoch_loss = (stats[0] / stats[3]).item()
            epoch_accuracy = (stats[1] / stats[2]).item()

            metrics = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
            }
            ray.train.report(metrics)

            if world_rank == 0:
                print(f"Epoch {epoch + 1}: loss = {epoch_loss:.4f} | accuracy = {epoch_accuracy:.4f}")
                
        # ----------------------------------------------------
        # training loop end
        # ----------------------------------------------------
        
        # ----------------------------------------------------
        # val loop start
        # ----------------------------------------------------
        
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle = False)
        
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                eeg = batch['eeg']
                instructions = batch['instruction']
                target_action_token_ids = batch["target_action_token_ids"]
                
                outputs = model(
                    eeg=eeg,
                    instructions=instructions
                )
                
                prediction_ids = outputs["predicted_action_token_ids"]
                target_action_token_ids = target_action_token_ids.to(device)
                matches = prediction_ids == target_action_token_ids # (B, 3) of boolean values
                
                total += (len(instructions) * 3)
                correct += matches.sum().item()
                
        eval_stats = torch.tensor([correct, total], device=device, dtype=torch.float64)
        eval_accuracy = (eval_stats[0] / eval_stats[1]).item()
        
        return {
            "eval_accuracy": eval_accuracy,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
        }
        
        # ----------------------------------------------------
        # val loop end
        # ----------------------------------------------------
        
    
    return train_and_val_loop_per_worker

def main():
    
    if RUN_SIM_VAL:
        simulation_validator = SimulationValidator(
            gen_variants_fn=gen_instruction_variants,
            flight_command_to_tokens=FLIGHT_COMMAND_TO_TOKENS,
            sim_manifest_path=SIM_MANIFEST_PATH,
        )
        simulation_validator.run()
        return
    
    else:
        
        accuracy_table = {
            "accuracy": [],
            "subject_id": [],
            "seed": [],
        }
    
        instruction_variations, flight_control_mappings = gen_instruction_variants()
        
        for seed in SEEDS:
        
            for subject, weights_path in zip(SUBJECTS, WEIGHT_PATHS):
                
                print("\n")
                print(f"SUBJECT {subject}. SEED {seed}")
                print("-----------------------------------------------------------")
                print("\n")
                
                trainer = TorchTrainer(
                    train_loop_per_worker=gen_train_func_for_ray_subject_and_seed(
                        seed,
                        subject,
                        weights_path,
                        EEGInceptionEncoder,
                        BLA,
                        gen_datasets,
                        DATA_PATH,
                        ALL_FILES,
                        instruction_variations,
                        flight_control_mappings
                    ),
                    train_loop_config={
                        "lr": LR,
                        "batch_size": BATCH_SIZE,
                        "epochs": NUM_EPOCHS
                    },
                    scaling_config=ScalingConfig(
                        num_workers=NUM_GPUS,
                        use_gpu=True
                    )
                )
                result = trainer.fit()
                acc = result.return_value["eval_accuracy"]
                
                accuracy_table["accuracy"].append(acc)
                accuracy_table["subject_id"].append(subject)
                accuracy_table['seed'].append(seed)
        
        accuracy_df = pd.DataFrame(accuracy_table)
        
        print("full accuracy breakdown:")
        print("\n")
        print(accuracy_df)
        
        metrics_dir = os.path.dirname(METRICS_PATH)
        os.makedirs(metrics_dir, exist_ok=True)

        accuracy_df.to_csv(METRICS_PATH, index=False)
        print(f"saved accuracy csv to {METRICS_PATH}")

if __name__ == "__main__":
    main()
