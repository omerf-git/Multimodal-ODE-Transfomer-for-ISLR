import torch
from torch import nn

from .common import FeatureExtractor, LinearClassifier


class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std


class VTNHCPF(nn.Module):
    def __init__(self, num_classes=226, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, **kwargs):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Transformer parameters
        d_model = 1024
        n_heads = 8
        num_layers = 4
        dim_feedforward = 2048
        transformer_dropout = 0.1  # You can adjust this

        self.feature_extractor = FeatureExtractor(cnn, 512, freeze_layers)  # Keep original embed_size for feature extractor

        # Project to transformer dimension
        input_features = 106 + 2 * 512  # pose_clip + 2 * embed_size
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(input_features, d_model)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, mm_clip):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = mm_clip

        # Reshape to put both hand crops on the same axis.
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        z = self.feature_extractor(rgb_clip)
        # Reshape back to extract features of both wrist crops as one feature vector.
        z = z.view(b, t, -1)

        zp = torch.cat((z, pose_clip), dim=-1)

        zp = self.norm(zp)
        zp = torch.nn.functional.relu(self.bottle_mm(zp), inplace=False)

        # Apply transformer encoder
        zp = self.transformer_encoder(zp)

        y = self.classifier(zp)

        return y.mean(1)
