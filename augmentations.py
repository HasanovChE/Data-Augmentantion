import torch
from torchvision import transforms
from PIL import Image

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.10):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            tensor = transforms.ToTensor()(img)
        else:
            tensor = img
            
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        
        if is_pil:
            return transforms.ToPILImage()(tensor)
        return tensor

def rotation_only():
    return [
        transforms.RandomRotation(degrees=15)
    ]

def rotation_shift():
    return [
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10))
    ]

def rotation_shift_noise():
    return [
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10)),
        AddGaussianNoise(std=0.10)
    ]