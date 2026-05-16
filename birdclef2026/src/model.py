"""BirdCLEF 2026 model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResidualDenoiser(nn.Module):
    """Multiplicative mask denoiser for mel spectrograms.

    Predicts a soft mask in [0, 1] and applies it element-wise:
        output = input * mask

    This is safer than additive residual (input - noise) because the mask
    can only suppress frequencies, never amplify or invent signal.
    Worst case: a frequency bin is zeroed out. It cannot over-subtract.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, 1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T) mel spectrogram

        Returns:
            denoised: (B, 1, n_mels, T)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        mask = torch.sigmoid(self.conv4(out))  # (B, 1, n_mels, T), values in [0, 1]
        return x * mask


class UNetDenoiser(nn.Module):
    """U-Net style denoiser with skip connections for mel spectrograms.
    
    Architecture:
        Encoder: 3 downsampling blocks (stride=2)
        Bottleneck: 2 conv blocks
        Decoder: 3 upsampling blocks with skip connections
        Output: Multiplicative mask in [0, 1]
    
    Args:
        base_channels: Base number of channels (default 32)
                      Total params ≈ 0.5M with base_channels=32
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(1, c)           # (B, c, H, W)
        self.enc2 = self._conv_block(c, c*2)         # (B, 2c, H/2, W/2)
        self.enc3 = self._conv_block(c*2, c*4)       # (B, 4c, H/4, W/4)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(c*4, c*8) # (B, 8c, H/8, W/8)
        
        # Decoder (upsampling path with skip connections)
        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3 = self._conv_block(c*8, c*4)       # c*8 because concat with enc3
        
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = self._conv_block(c*4, c*2)       # c*4 because concat with enc2
        
        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = self._conv_block(c*2, c)         # c*2 because concat with enc1
        
        # Output layer: predict mask
        self.out_conv = nn.Conv2d(c, 1, 1)
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Double convolution block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T) mel spectrogram

        Returns:
            denoised: (B, 1, n_mels, T)
        """
        # Encoder with skip connections
        e1 = self.enc1(x)              # (B, c, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 2c, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 4c, H/4, W/4)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 8c, H/8, W/8)
        
        # Decoder with skip connections
        # 需要處理尺寸不匹配的問題
        d3 = self.up3(b)                           # (B, 4c, ~H/4, ~W/4)
        # 調整 d3 的尺寸以匹配 e3
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)            # (B, 8c, H/4, W/4)
        d3 = self.dec3(d3)                         # (B, 4c, H/4, W/4)
        
        d2 = self.up2(d3)                          # (B, 2c, ~H/2, ~W/2)
        # 調整 d2 的尺寸以匹配 e2
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)            # (B, 4c, H/2, W/2)
        d2 = self.dec2(d2)                         # (B, 2c, H/2, W/2)
        
        d1 = self.up1(d2)                          # (B, c, ~H, ~W)
        # 調整 d1 的尺寸以匹配 e1
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)            # (B, 2c, H, W)
        d1 = self.dec1(d1)                         # (B, c, H, W)
        
        # Output: multiplicative mask
        mask = torch.sigmoid(self.out_conv(d1))    # (B, 1, H, W), values in [0, 1]
        
        # 確保輸出尺寸與輸入完全一致
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return x * mask


class AttentionPooling(nn.Module):
    """Simple attention pooling over spatial dimensions."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, C, T)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)  # (B, C, H*W)
        # x: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, C)
        return pooled


class BirdCLEFModel(nn.Module):
    """BirdCLEF classifier with optional denoising CNN + timm backbone.

    Args:
        backbone_name: timm model name (e.g. 'resnet18')
        num_classes: number of output classes (default 234)
        pretrained: whether to load ImageNet pretrained weights (default True)
        checkpoint_path: optional path to a local checkpoint file for backbone weights
        pool: pooling strategy, 'avg' or 'attention'
        use_denoiser: whether to use denoiser before classification
        denoiser_channels: number of channels in denoiser CNN
        denoiser_type: type of denoiser ('residual' or 'unet')
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 234,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        pool: str = "avg",
        use_denoiser: bool = False,
        denoiser_channels: int = 64,
        denoiser_type: str = "residual",
    ):
        super().__init__()

        # Optional denoising CNN
        self.use_denoiser = use_denoiser
        if use_denoiser:
            if denoiser_type == "unet":
                self.denoiser = UNetDenoiser(base_channels=denoiser_channels)
            else:  # default to residual
                self.denoiser = ResidualDenoiser(channels=denoiser_channels)

        # Create backbone without classification head
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0
        )

        # Load pretrained weights from local file if provided
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Handle various checkpoint formats
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
            self.backbone.load_state_dict(state_dict, strict=False)
        elif pretrained:
            # Load ImageNet pretrained weights via timm
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, num_classes=0
            )

        # Determine feature dimension
        feature_dim = self.backbone.num_features

        # Pooling
        self.pool_type = pool
        if pool == "attention":
            self.pooling = AttentionPooling(feature_dim)
        elif pool == "mean_max":
            # Concatenate global mean and global max → double feature dim
            self.pooling = None
            feature_dim = feature_dim * 2
        else:
            self.pooling = None  # will use adaptive avg pool

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (B, 1, n_mels, T)

        Returns:
            logits: tensor of shape (B, num_classes), NO sigmoid applied
        """
        # Optional denoising step
        if self.use_denoiser:
            x = self.denoiser(x)  # (B, 1, n_mels, T)

        # Expand single-channel to 3 channels for ImageNet-pretrained backbones
        x = x.repeat(1, 3, 1, 1)  # (B, 3, n_mels, T)

        # Extract features via backbone
        features = self.backbone.forward_features(x)  # (B, C, H, W) or (B, C)

        # Apply pooling
        if self.pool_type == "attention":
            if features.dim() == 2:
                pooled = features
            else:
                pooled = self.pooling(features)
        elif self.pool_type == "mean_max":
            if features.dim() == 4:
                mean_pool = features.mean(dim=[2, 3])  # (B, C)
                max_pool  = features.amax(dim=[2, 3])  # (B, C)
            elif features.dim() == 3:
                mean_pool = features.mean(dim=1)
                max_pool  = features.amax(dim=1)
            else:
                mean_pool = max_pool = features
            pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2C)
        else:
            # Global average pooling
            if features.dim() == 4:
                pooled = features.mean(dim=[2, 3])
            elif features.dim() == 3:
                pooled = features.mean(dim=1)
            else:
                pooled = features

        # Classification head — raw logits, no sigmoid
        logits = self.classifier(pooled)
        return logits
