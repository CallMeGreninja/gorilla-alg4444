import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
import numpy as np
from dataset import MonkeyDataset
from utils import collate_fn
from tqdm import tqdm

NUM_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-5

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
    )
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    
    for pas_patches, target_heatmaps in val_loader:
        pas_patches = pas_patches.to(device)
        target_heatmaps = target_heatmaps.to(device)
        
        predicted_heatmaps = model(pas_patches)
        loss = criterion(predicted_heatmaps, target_heatmaps)
        
        total_val_loss += loss.item()
        
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

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
    
    if os.path.exists(best_model_path):
        print(f"🔄 Loading best weights from {best_model_path} to resume training...")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = np.inf

    print(f"Starting training. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}.")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for pas_patches, target_heatmaps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
            pas_patches = pas_patches.to(DEVICE)
            target_heatmaps = target_heatmaps.to(DEVICE)
            
            optimizer.zero_grad()
            predicted_heatmaps = model(pas_patches)
            loss = criterion(predicted_heatmaps, target_heatmaps)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)

        epoch_val_loss = validate_model(model, val_loader, criterion, DEVICE)
        
        print(f"\nEpoch {epoch+1} finished. Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model improved! Saving best model to {save_path}")
        
if __name__ == '__main__':
    if DEVICE.type == 'cuda':
        print(f"CUDA is being used! Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training on CPU")

    print("--- Initializing Train Dataset (Annotations/ROI loading happens now) ---")
    train_dataset = MonkeyDataset(data_root_dir=TRAIN_DATA_ROOT, split='train')
    print("--- Initializing Validation Dataset (Annotations/ROI loading happens now) ---")
    val_dataset = MonkeyDataset(data_root_dir=VAL_DATA_ROOT, split='val')

    train_model(train_dataset, val_dataset)