import torch
import torch.nn as nn
from encoder_classifiers import ClassifierModule


class EEGFlattenedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 22,       # 22 electrodes
        seq_len: int = 875,        # 3.5 seconds at 250 Hz: 2.5s -> 6.0s
        patch_len: int = 175,      # 5 patches, each 175 timesteps
        num_classes: int = 4,      # left hand, right hand, feet, tongue
        patch_feature_dim: int = 16,
        d_model: int = 128,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.patch_feature_dim = patch_feature_dim
        
        # each timestep has 22 electrodes
        # project each timestep's data into d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # positional embeddings across each of the full 875 timesteps
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model)
        )
        
        # each patch will have shape patch_len * d_model
        self.patch_flattened_dim = patch_len * d_model
        
        # same MLP is applied to each patch
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.patch_flattened_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        self.classifier_module = ClassifierModule(
            d_model = d_model,
            patch_feature_dim = self.patch_feature_dim,
            num_classes = num_classes,
            num_patches = self.num_patches
        )
        
    
    def encode(self, x: torch.Tensor):
        batch_size, seq_len, input_dim = x.shape
        
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim
        
        # (B, 875, 22) -> (B, 875, 128)
        x = self.input_projection(x)
        
        # add positional embeddings
        # (B, 875, 128)
        x = x + self.pos_embedding
        
        # split into patches
        # (B, 875, 128) -> (B, 5, 175, 128)
        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_len,
            self.d_model
        )
        
        # flatten each patch
        # (B, 5, 175, 128) -> (B, 5, 22400)
        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_flattened_dim
        )
        
        # (B, 5, 22400) -> (B, 5, 128)
        embeddings = self.patch_mlp(x)
        
        return embeddings
        
    
    def forward(self, x: torch.Tensor):
        
        # (B, 875, 22) -> (B, 5, 128)
        embeddings = self.encode(x)
        
        # (B, 5, 128) -> (B, 4)
        logits = self.classifier_module(embeddings)
        
        return logits