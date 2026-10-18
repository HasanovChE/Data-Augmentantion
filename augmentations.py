import torch
from torchvision import transforms

# Rotation
def rotation_only():
    return [
        transforms.RandomRotation(
            degrees=15
        )
    ]

# Rotation + Shift
def rotation_shift():
    return [
        transforms.RandomRotation(
            degrees=15
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(
                0.10,
                0.10
            )
        )
    ]

# Rotation + Shift + Noise
class AddGaussianNoise:
    def __init__(
        self,
        mean=0.0,
        std=0.10
    ):
        self.mean = mean
        self.std = std
        
    def __call__(
        self,
        tensor
    ):
        noise = torch.randn_like(
            tensor
        ) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(
            tensor,
            -3.0,
            3.0
        )

def rotation_shift_noise():
    return [
        transforms.RandomRotation(
            degrees=15
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(
                0.10,
                0.10
            )
        ),
        AddGaussianNoise(
            std=0.10
        )
    ]

# Configuration dictionary
CONFIGURATIONS = {
    "baseline": None,
    "rotation": rotation_only(),
    "rotation_shift": rotation_shift(),
    "rotation_shift_noise": rotation_shift_noise()
}