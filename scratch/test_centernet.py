import torch
from src.dataset import ForamsDataset
from src.model import HybridForamsModel

def test():
    dataset = ForamsDataset(data_dir="data/synthetic/train")
    # Take a single sample
    img, targets = dataset[0]
    
    print(f"Image shape: {img.shape}")
    print(f"Heatmap target shape: {targets['heatmap'].shape}")
    print(f"Offset target shape: {targets['offset'].shape}")
    print(f"Mask target shape: {targets['mask'].shape}")
    
    # Add batch dim
    img = img.unsqueeze(0)
    targets = {k: v.unsqueeze(0) for k, v in targets.items()}
    
    model = HybridForamsModel()
    
    # Forward pass
    hm, off = model(img)
    print(f"Heatmap pred shape: {hm.shape}")
    print(f"Offset pred shape: {off.shape}")
    
    # Test loss
    batch = (img, targets)
    loss = model.training_step(batch, 0)
    print(f"Training step loss: {loss.item()}")

if __name__ == "__main__":
    test()
