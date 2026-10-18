from torchvision import datasets
from torchvision import transforms

def get_datasets(augmentation=None):
    base_transform = [
        transforms.ToTensor()
    ]
    
    if augmentation is not None:
        train_transform = transforms.Compose(base_transform + augmentation)
    else:
        train_transform = transforms.Compose(base_transform)
        
    test_transform = transforms.Compose(base_transform)
    
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )
    
    return train_dataset, test_dataset