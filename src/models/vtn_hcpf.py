import argparse
import torch
from torch import nn

from .common import FeatureExtractor, LinearClassifier, PositionEncoding
from .ode_transformer_encoder import ODETransformerEncoder

class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std


class VTNHCPF(nn.Module):
    def __init__(self, num_classes=226, num_heads=8, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 freeze_layers=0, dropout=0, norm_first=True,  enc_calculate_num=1, rk_type="none", encoder_history_type=None, **kwargs):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.norm_first = norm_first
        self.enc_calculate_num = enc_calculate_num
        self.rk_type = rk_type
        self.encoder_history_type = encoder_history_type
        # Transformer parameters
        d_model = embed_size * 2  # You can adjust this
        dim_feedforward = d_model * 2
        transformer_dropout = dropout  # You can adjust this

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)  # Keep original embed_size for feature extractor
        num_attn_features = embed_size * 2
        # Project to transformer dimension
        input_features = 106 + 2 * embed_size  # pose_clip + 2 * embed_size
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(input_features, d_model)
        self.position_encoder = PositionEncoding(self.sequence_length,num_attn_features)  # Not using positional encoding for now

        if self.norm_first:
            normalization = nn.LayerNorm(d_model)
        else:
            normalization = None
        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation='relu',
            batch_first=True,
            norm_first=norm_first
        )
        history_args = argparse.Namespace(
            encoder_embed_dim=d_model,
            decoder_embed_dim=d_model,
            encoder_history_type=self.encoder_history_type,
            decoder_history_type=None,
            encoder_normalize_before=self.norm_first,
            decoder_normalize_before=True,
            encoder_layers=num_layers,
            decoder_layers=0,
            encoder_integration_type="avg",
        )
        self.transformer_encoder = ODETransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=normalization,
            rk_type=self.rk_type,
            calculate_num=self.enc_calculate_num,
            history_args=history_args
        )
        
        self.classifier = LinearClassifier(num_attn_features, num_classes)

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
        zp = self.position_encoder(zp)
        # Apply transformer encoder
        zp = self.transformer_encoder(zp)

        y = self.classifier(zp)

        return y.mean(1)
