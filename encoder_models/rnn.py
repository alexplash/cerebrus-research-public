import torch
import torch.nn as nn
from encoder_classifiers import ClassifierModule


class EEGRNNEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 22,   # 22 electrodes
        seq_len: int = 875,    # 3.5 seconds at 250 Hz: 2.5s -> 6.0s
        patch_len: int = 175,  # 5 patches per EEG window
        num_classes: int = 4,  # left hand, right hand, feet, tongue
        patch_feature_dim: int = 16,
        d_model: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.num_layers = num_layers
        self.patch_feature_dim = patch_feature_dim
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.classifier_module = ClassifierModule(
            d_model = self.d_model,
            patch_feature_dim=self.patch_feature_dim,
            num_classes=self.num_classes,
            num_patches=self.num_patches
        )
    
    def encode(self, x: torch.Tensor):
        
        batch_size, seq_len, input_dim = x.shape
        
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim
        
        # (B, 875, 22) -> (B, 875, 128)
        x = self.input_projection(x)
        
        # split into patches
        # (B, 875, 128) -> (B, 5, 175, 128)
        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_len,
            self.d_model
        )
        
        # treat each patch as a separate sequence
        # (B, 5, 175, 128) -> (B * 5, 175, 128)
        x = x.reshape(
            batch_size * self.num_patches,
            self.patch_len,
            self.d_model
        )
        
        # run each patch through RNN
        # output: (B * 5, 175, 128)
        # h_n: (num_layers, B * 5, 128)
        output, h_n = self.rnn(x)
        
        # use final hidden state from final RNN layer as the patch embedding
        # (num_layers, B * 5, 128) -> (B * 5, 128)
        patch_embedding = h_n[-1]
        
        # (B * 5, 128) -> (B, 5, 128)
        embeddings = patch_embedding.reshape(
            batch_size,
            self.num_patches,
            self.d_model
        )
        
        return embeddings
    
    def forward(self, x: torch.Tensor):
        
        # (B, 875, 22) -> (B, 5, 128)
        embeddings = self.encode(x)
        
        # (B, 5, 128) -> (B, 4)
        logits = self.classifier_module(embeddings)
        
        return logits