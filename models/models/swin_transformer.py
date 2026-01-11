"""
swin_transformer.py

Loads a pretrained Swin Transformer model for deepfake detection.
"""

import torch
import timm  # PyTorch Image Models library


def load_swin(pretrained=True):
    """
    Loads a Swin Transformer model in evaluation mode.

    Args:
        pretrained (bool): If True, load pretrained weights

    Returns:
        model (torch.nn.Module): Pretrained Swin Transformer
    """
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    model.eval()
    return model
