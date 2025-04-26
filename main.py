from dataset import TrackingNetDataset
from model import SimpleObjectTracker
from train import train_SimpleObjectTracker
from utils import denormalize

import os
import torch
from torch.utils.data import DataLoader

root_dir = 'TrackingNet/TRAIN_0'
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"Directory '{root_dir}' does not exist.")

dataset = TrackingNetDataset(root_dir=root_dir)
model = SimpleObjectTracker()
train_SimpleObjectTracker(model, dataset, batch_size=64, num_epochs=50, learning_rate=0.001, save_path='simple_tracker.pth')
