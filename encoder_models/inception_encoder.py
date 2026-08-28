
import torch.nn as nn
import torch
from encoder_classifiers import ClassifierModule

class EEGInceptionEncoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 22,
        num_classes: int = 4,
        seq_len: int = 875,
        patch_len: int = 175,
        d_model: int = 128,
        patch_feature_dim: int = 16,
        sfreq=250,
        n_filters: int = 48,
        kernel_unit_s: float = 0.1,
        inception_kernel_schedule: list[int] = [25, 51, 75, 125, 175],
        activation: type[nn.Module] = nn.ReLU,
    ):
        
        super().__init__()
        
        self.n_filters = n_filters
        self.kernel_unit_s = kernel_unit_s
        self.activation = activation
        
        self.inception_kernel_schedule = inception_kernel_schedule
        self.n_convs = len(self.inception_kernel_schedule)
        
        self.input_dim = input_dim
        self.sfreq = sfreq
        self.num_classes = num_classes
        self.seq_len = seq_len
        
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.d_model = d_model
        self.patch_feature_dim = patch_feature_dim
        
        self.initial_inception_module = InceptionModule(
            in_channels=self.input_dim,
            n_filters=self.n_filters,
            n_convs=self.n_convs,
            kernel_unit_s=self.kernel_unit_s,
            sfreq=self.sfreq,
            activation=self.activation,
            inception_kernel_schedule=self.inception_kernel_schedule
        )

        self.intermediate_in_channels = (self.n_convs + 1) * self.n_filters # 288
        
        self.intermediate_inception_modules_1 = nn.ModuleList(
            [
                InceptionModule(
                    in_channels=self.intermediate_in_channels,
                    n_filters=self.n_filters,
                    n_convs=self.n_convs,
                    kernel_unit_s=self.kernel_unit_s,
                    sfreq=self.sfreq,
                    activation=self.activation,
                    inception_kernel_schedule=self.inception_kernel_schedule
                )
                for _ in range(2)
            ]
        )
        
        self.residual_block_1 = ResidualModule(
            in_channels=self.input_dim,
            n_filters=self.intermediate_in_channels,
            activation=self.activation,
        )
        
        self.intermediate_inception_modules_2 = nn.ModuleList(
            [
                InceptionModule(
                    in_channels=self.intermediate_in_channels,
                    n_filters=self.n_filters,
                    n_convs=self.n_convs,
                    kernel_unit_s=self.kernel_unit_s,
                    sfreq=self.sfreq,
                    activation=self.activation,
                    inception_kernel_schedule=self.inception_kernel_schedule
                )
                for _ in range(3)
            ]
        )
        
        self.residual_block_2 = ResidualModule(
            in_channels=self.intermediate_in_channels,
            n_filters=self.intermediate_in_channels,
            activation=self.activation,
        )
        
        self.ave_pooling = nn.AvgPool2d(
            kernel_size=(1, self.patch_len),
        )
        
        self.embedding_projection = nn.Linear(
            self.intermediate_in_channels,
            self.d_model
        )
        
        self.flat = nn.Flatten()
        
        self.identity = nn.Identity()
        
        self.classifier_module = ClassifierModule(
            d_model = self.d_model,
            patch_feature_dim = self.patch_feature_dim,
            num_classes=self.num_classes,
            num_patches=self.num_patches
        )
        
    
    def encode(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
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
        
        # (B * 5, 175, 22) -> (B * 5, 22, 175)
        x = x.permute(0, 2, 1).contiguous()
        
        # (B * 5, 22, 175) -> (B * 5, 22, 175, 1)
        x = x.unsqueeze(-1)
        
        # (B * 5, 22, 175, 1) -> (B * 5, 22, 1, 175)
        x = x.permute(0, 1, 3, 2)
        
        # (B * 5, 22, 1, 175) -> (B * 5, 288, 1, 175)
        res1 = self.residual_block_1(x)
        
        # (B * 5, 22, 1, 175) -> (B * 5, 288, 1, 175)
        x = self.initial_inception_module(x)
        for layer in self.intermediate_inception_modules_1:
            # (B * 5, 288, 1, 175) -> (B * 5, 288, 1, 175)
            x = layer(x)
        
        # (B * 5, 288, 1, 175) + (B * 5, 288, 1, 175) -> (B * 5, 288, 1, 175)
        x = x + res1
        
        # (B * 5, 288, 1, 175) -> (B * 5, 288, 1, 175)
        res2 = self.residual_block_2(x)
        
        for layer in self.intermediate_inception_modules_2:
            # (B * 5, 288, 1, 175) -> (B * 5, 288, 1, 175)
            x = layer(x)
        
        # (B * 5, 288, 1, 175) + (B * 5, 288, 1, 175) -> (B * 5, 288, 1, 175)
        x = res2 + x
        
        # (B * 5, 288, 1, 175) -> (B * 5, 288, 1, 1)
        x = self.ave_pooling(x)
        
        # (B * 5, 288, 1, 1) -> (B * 5, 288)
        x = self.flat(x)
        
        # (B * 5, 288) -> (B * 5, 128)
        patch_embeddings = self.embedding_projection(x)
        
        # (B * 5, 128) -> (B * 5, 128)
        patch_embeddings = self.identity(patch_embeddings)
        
        # (B * 5, 128) -> (B, 5, 128)
        embeddings = patch_embeddings.reshape(
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



class InceptionModule(nn.Module):

    def __init__(
        self,
        in_channels,
        n_filters,
        n_convs,
        inception_kernel_schedule,
        kernel_unit_s=0.1,
        sfreq=250,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_convs = n_convs
        self.kernel_unit_s = kernel_unit_s
        self.sfreq = sfreq
        
        self.inception_kernel_schedule = inception_kernel_schedule

        self.bottleneck = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_filters,
            kernel_size=1,
            bias=True,
        )

        kernel_unit = int(self.kernel_unit_s * self.sfreq)

        self.pooling = nn.MaxPool2d(
            kernel_size=(1, kernel_unit),
            stride=1,
            padding=(0, int(kernel_unit // 2)),
        )

        self.pooling_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_filters,
            kernel_size=1,
            bias=True,
        )
        
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=(1, size),
                    padding="same",
                    bias=True,
                )
                for size in self.inception_kernel_schedule
            ]
        )

        self.bn = nn.BatchNorm2d(self.n_filters * (self.n_convs + 1))

        self.activation = activation()

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        
        X1 = self.bottleneck(X)

        X1 = [conv(X1) for conv in self.conv_list]

        X2 = self.pooling(X)
        X2 = self.pooling_conv(X2)
        # Get the target length from one of the conv branches
        target_len = X1[0].shape[-1]

        # Crop the pooling output if its length does not match
        if X2.shape[-1] != target_len:
            X2 = X2[..., :target_len]

        out = torch.cat(X1 + [X2], 1)

        out = self.bn(out)
        return self.activation(out)



class ResidualModule(nn.Module):

    def __init__(self, in_channels, n_filters, activation: type[nn.Module] = nn.ReLU):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.activation = activation()

        self.bn = nn.BatchNorm2d(self.n_filters)
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_filters,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        
        out = self.conv(X)
        out = self.bn(out)
        return self.activation(out)
    

        
        
