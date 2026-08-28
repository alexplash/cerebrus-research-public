import torch
import torch.nn as nn
from encoder_classifiers import ClassifierModule


class EEGSubtractiveTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 22,        # 22 electrodes
        seq_len: int = 875,         # 3.5 seconds at 250 Hz: 2.5s -> 6.0s
        patch_len: int = 175,       # 5 patches per EEG window
        num_classes: int = 4,       # left hand, right hand, feet, tongue
        patch_feature_dim: int = 16,
        d_model: int = 128,
        num_heads: int = 8,
        feed_hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.num_classes = num_classes
        self.d_model = d_model
        self.patch_feature_dim = patch_feature_dim

        # Project each timestep of 22 electrodes into d_model.
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional embedding for each timestep and each d_model position.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model)
        )

        # One learned CLS token used for each patch.
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        encoder_layer_A = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feed_hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder_A = nn.TransformerEncoder(
            encoder_layer=encoder_layer_A,
            num_layers=num_layers,
        )
        
        encoder_layer_B = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feed_hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder_B = nn.TransformerEncoder(
            encoder_layer=encoder_layer_B,
            num_layers=num_layers,
        )
        
        self.subtractive_lambda = nn.Parameter(torch.tensor(0.5))
        
        self.classifier_module = ClassifierModule(
            d_model=self.d_model,
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

        # add positional embeddings
        # (B, 875, 128)
        x = x + self.pos_embedding

        # split into patches
        # (B, 875, 128) -> (B, 5, 175, 128)
        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_len,
            self.d_model,
        )

        # treat each patch as its own independent sequence
        # (B, 5, 175, 128) -> (B * 5, 175, 128)
        x = x.reshape(
            batch_size * self.num_patches,
            self.patch_len,
            self.d_model,
        )

        # create one CLS token for each patch sequence
        # (1, 1, 128) -> (B * 5, 1, 128)
        cls_tokens = self.cls_token.expand(
            batch_size * self.num_patches,
            -1,
            -1,
        )

        # prepend CLS token to each patch sequence
        # (B * 5, 175, 128) -> (B * 5, 176, 128)
        x = torch.cat([cls_tokens, x], dim=1)

        # apply the subtractive transformer encoder to each patch independently
        # (B * 5, 176, 128) -> (B * 5, 176, 128)
        A = self.transformer_encoder_A(x)
        B = self.transformer_encoder_B(x)
        x = A - self.subtractive_lambda * B

        # use CLS output as the embedding for each patch
        # (B * 5, 176, 128) -> (B * 5, 128)
        patch_embeddings = x[:, 0, :]

        # restore patch structure
        # (B * 5, 128) -> (B, 5, 128)
        embeddings = patch_embeddings.reshape(
            batch_size,
            self.num_patches,
            self.d_model,
        )

        return embeddings

    def forward(self, x: torch.Tensor):
        
        # (B, 875, 22) -> (B, 5, 128)
        embeddings = self.encode(x)
        
        # (B, 5, 128) -> (B, 4)
        logits = self.classifier_module(embeddings)
        
        return logits