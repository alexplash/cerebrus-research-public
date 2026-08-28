import torch
import torch.nn as nn
from encoder_classifiers import ClassifierModule


class EEGSubtractiveConv2D(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 22,   # 22 electrodes
        seq_len: int = 875,    # 3.5 seconds at 250 Hz
        patch_len: int = 175,  # 5 patches per EEG window
        num_classes: int = 4,  # left hand, right hand, feet, tongue
        patch_feature_dim: int = 16,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
    ):
        super().__init__()
        
        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.num_classes = num_classes
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.patch_feature_dim = patch_feature_dim
        
        self.conv1_A = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_channels,
                kernel_size=(5, input_dim),
                padding=(2, 0)
            ),
            nn.GELU()
        )
        
        self.conv1_B = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_channels,
                kernel_size=(5, input_dim),
                padding=(2, 0)
            ),
            nn.GELU()
        )
        
        self.conv2_A = nn.Sequential(
            nn.Conv2d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=(5, 1),
                padding=(2, 0)
            ),
            nn.GELU()
        )
        
        self.conv2_B = nn.Sequential(
            nn.Conv2d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=(5, 1),
                padding=(2, 0)
            ),
            nn.GELU()
        )
        
        self.lambda1 = nn.Parameter(torch.tensor(0.5))
        self.lambda2 = nn.Parameter(torch.tensor(0.5))
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier_module = ClassifierModule(
            d_model=self.conv2_channels,
            patch_feature_dim=self.patch_feature_dim,
            num_classes=self.num_classes,
            num_patches=self.num_patches
        )
    
    def encode(self, x: torch.Tensor):
        
        batch_size, seq_len, input_dim = x.shape
        
        assert seq_len == self.seq_len
        assert input_dim == self.input_dim
        
        # split into patches
        # (B, 875, 22) -> (B, 5, 175, 22)
        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_len,
            self.input_dim
        )
        
        # treat each patch as its own independent sequence
        # (B, 5, 175, 22) -> (B * 5, 175, 22)
        x = x.reshape(
            batch_size * self.num_patches,
            self.patch_len,
            self.input_dim
        )
        
        # (B * 5, 175, 22) -> (B * 5, 1, 175, 22)
        x = x.unsqueeze(1)
        
        # (B * 5, 1, 175, 22) -> (B * 5, 64, 175, 1)
        A = self.conv1_A(x)
        B = self.conv1_B(x)
        x = A - self.lambda1 * B
        
        # (B * 5, 64, 175, 1) -> (B * 5, 128, 175, 1)
        A = self.conv2_A(x)
        B = self.conv2_B(x)
        x = A - self.lambda2 * B
        
        # (B * 5, 128, 175, 1) -> (B * 5, 128, 1, 1)
        x = self.pool(x)
        
        # (B * 5, 128, 1, 1) -> (B * 5, 128)
        x = x.reshape(batch_size * self.num_patches, -1)
        
        # (B * 5, 128) -> (B, 5, 128)
        embeddings = x.reshape(
            batch_size,
            self.num_patches,
            self.conv2_channels,
        )
        
        return embeddings
    
    def forward(self, x: torch.Tensor):
        
        # (B, 875, 22) -> (B, 5, 128)
        embeddings = self.encode(x)
        
        # (B, 5, 128) -> (B, 4)
        logits = self.classifier_module(embeddings)
        
        return logits