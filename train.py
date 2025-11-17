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
TRAIN_DATA_ROOT = './data/train'
VAL_DATA_ROOT = './data/validation'
MODEL_SAVE_DIR = './model_weights'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Debugging
if DEVICE.type == 'cuda':
    print(f"CUDA is being used! Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Training on CPU")

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