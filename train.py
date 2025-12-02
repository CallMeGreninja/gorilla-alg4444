import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
import numpy as np
from dataset import MonkeyDataset
from utils import collate_fn
from tqdm import tqdm

NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_DIR = './model_weights'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

TRAIN_DATA_ROOT = './data/train'
VAL_DATA_ROOT = './data/validation'

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation='sigmoid',
    )
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (pas_patches, target_heatmaps) in enumerate(val_loader):
            pas_patches = pas_patches.to(device)
            target_heatmaps = target_heatmaps.to(device)
            
            predicted_heatmaps = model(pas_patches)
            
            if batch_idx == 0:
                print(f"Val - Max Pred: {predicted_heatmaps.max().item():.4f}, "
                      f"Mean Pred: {predicted_heatmaps.mean().item():.4f}")

            loss = criterion(predicted_heatmaps, target_heatmaps)
            total_val_loss += loss.item()
        
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=100.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        weights = torch.ones_like(target) + (target > 0.05).float() * (self.weight - 1)
        loss = loss * weights
        return loss.mean()
    
def train_model(train_dataset, val_dataset):    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )

    model = build_model().to(DEVICE)

    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    criterion = WeightedMSELoss(weight=100.0) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    best_val_loss = np.inf

    print(f"Starting training. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}.")
    print(f"Training for {NUM_EPOCHS} epochs with LR={LEARNING_RATE}")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (pas_patches, target_heatmaps) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            pas_patches = pas_patches.to(DEVICE)
            target_heatmaps = target_heatmaps.to(DEVICE)

            if epoch == 0 and batch_idx == 0:
                with torch.no_grad():
                    sample_pred = model(pas_patches)
                    print(f"\n🔍 First batch check:")
                    print(f"   Pred range: [{sample_pred.min().item():.4f}, {sample_pred.max().item():.4f}]")
                    print(f"   Pred mean: {sample_pred.mean().item():.4f}")
                    print(f"   Target range: [{target_heatmaps.min().item():.4f}, {target_heatmaps.max().item():.4f}]")
                    print(f"   Target has cells: {(target_heatmaps > 0.05).sum().item()} pixels\n")
            
            optimizer.zero_grad()
            predicted_heatmaps = model(pas_patches)
            loss = criterion(predicted_heatmaps, target_heatmaps)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)

        epoch_val_loss = validate_model(model, val_loader, criterion, DEVICE)

        scheduler.step()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {epoch_train_loss:.6f}")
        print(f"  Val Loss:   {epoch_val_loss:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model improved! Val loss: {best_val_loss:.6f}\n")
    
    print(f"\n🎉 Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {best_model_path}")
        
if __name__ == '__main__':
    if DEVICE.type == 'cuda':
        print(f"✅ CUDA available! Training on: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Training on CPU (will be slow)")

    print("\n--- Initializing Train Dataset ---")
    train_dataset = MonkeyDataset(data_root_dir=TRAIN_DATA_ROOT, split='train')
    print("--- Initializing Validation Dataset ---")
    val_dataset = MonkeyDataset(data_root_dir=VAL_DATA_ROOT, split='val')

    train_model(train_dataset, val_dataset)