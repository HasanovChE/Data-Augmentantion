import torch

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, prediction, y_a, y_b, lam):
    return lam * criterion(prediction, y_a) + (1 - lam) * criterion(prediction, y_b)