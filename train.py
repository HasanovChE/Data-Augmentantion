import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNN
from utils import set_seed

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

def create_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_model(train_loader, val_loader, epochs=10, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = []
    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
        train_loss = total_loss / total
        train_accuracy = correct / total
        val_accuracy = evaluate(model, val_loader, device)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })
        
        print(
            f"Epoch {epoch+1}/{epochs} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}"
        )
        
    training_time = time.perf_counter() - start_time
    return model, history, training_time