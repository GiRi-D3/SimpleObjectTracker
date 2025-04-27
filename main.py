from dataset import TrackingNetDataset
from model import SimpleObjectTracker
from train import train_SimpleObjectTracker
from utils import denormalize
from config import MODEL_CONFIG, TRAIN_CONFIG

import os
import torch
from torch.utils.data import DataLoader

root_dir = 'TrackingNet/TRAIN_0'
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"Directory '{root_dir}' does not exist.")

dataset = TrackingNetDataset(root_dir=root_dir)

model = SimpleObjectTracker()

train_SimpleObjectTracker(
    model, 
    dataset,
    batch_size=TRAIN_CONFIG['batch_size'],
    num_epochs=TRAIN_CONFIG['num_epochs'],
    learning_rate=TRAIN_CONFIG['learning_rate'],
    save_path=TRAIN_CONFIG['save_path']
)
