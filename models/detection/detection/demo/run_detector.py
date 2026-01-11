"""
run_detector.py

Demo script for DeepTrace deepfake detection.
"""

import torch
from detection.deepfake_detector import detect_deepfake

def main():
    # Simulate a dummy image tensor (1 x 3 x 224 x 224)
    image_tensor = torch.rand((1, 3, 224, 224))

    deepfake_prob = detect_deepfake(image_tensor)
    print(f"Deepfake probability: {deepfake_prob:.4f}")

if __name__ == "__main__":
    main()
