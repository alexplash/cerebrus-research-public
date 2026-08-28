
import torch
import torch.nn as nn

class ClassifierModule(nn.Module):
    
    def __init__(
        self,
        d_model,
        patch_feature_dim,
        num_classes,
        num_patches
    ):
        
        super().__init__()
        
        self.d_model = d_model
        self.patch_feature_dim = patch_feature_dim
        self.num_classes = num_classes
        self.num_patches = num_patches
        
        self.patch_projection = nn.Sequential(
            nn.Linear(self.d_model, self.patch_feature_dim),
            nn.GELU()
        )
        
        self.classifier_head = nn.Linear(
            self.num_patches * self.patch_feature_dim, 
            num_classes
        )
    
    def forward(self, x:torch.Tensor):
        
        assert len(x.shape) == 3
        assert x.shape[1] == self.num_patches
        assert x.shape[2] == self.d_model
        
        # (B, 5, 128) -> (B, 5, 16)
        patch_features = self.patch_projection(x)
        
        # (B, 5, 16) -> (B, 80)
        batch_size = patch_features.shape[0]
        flattened_features = patch_features.reshape(
            batch_size,
            patch_features.shape[1] * patch_features.shape[2]
        )
        
        # (B, 80) -> (B, 4)
        logits = self.classifier_head(flattened_features)
        
        return logits