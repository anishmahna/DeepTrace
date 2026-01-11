"""
deepfake_detector.py

Deepfake detection pipeline using Swin Transformer.
"""

import torch
from models.swin_transformer import load_swin
import torch.nn.functional as F


def detect_deepfake(image_tensor):
    """
    Runs deepfake detection on a single image tensor.

    Args:
        image_tensor (torch.Tensor): Preprocessed image (1 x 3 x 224 x 224)

    Returns:
        float: probability of image being a deepfake
    """
    model = load_swin()
    
    with torch.no_grad():
        output = model(image_tensor)
        prob = F.softmax(output, dim=1)
    
    # Assuming class 1 is deepfake
    deepfake_prob = prob[0][1].item()
    return deepfake_prob
