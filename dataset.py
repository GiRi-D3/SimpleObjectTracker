import os
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import DATA_CONFIG, IMG_CONFIG

class TrackingNetDataset(Dataset):
    def __init__(self, root_dir=DATA_CONFIG['root_dir'], 
                 seq_length=DATA_CONFIG['seq_length'], 
                 transform=None, 
                 img_size=DATA_CONFIG['img_size']):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_size = img_size
        self.frames_dir = os.path.join(root_dir, 'frames')
        self.anno_dir = os.path.join(root_dir, 'anno')
        self.video_ids = sorted(os.listdir(self.frames_dir))
        self.sequence_indices = self._build_sequence_indices()

    def _build_sequence_indices(self):
        sequence_indices = []
        for video_id in self.video_ids:
            frame_folder = os.path.join(self.frames_dir, video_id)
            frame_files = sorted(
                [f for f in os.listdir(frame_folder) if f.endswith('.jpg')],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            for start_idx in range(len(frame_files) - self.seq_length):
                sequence_indices.append((video_id, start_idx))
        return sequence_indices

    def _load_frame(self, video_id, frame_idx):
        frame_path = os.path.join(
            self.frames_dir, 
            video_id, 
            f"{frame_idx}.jpg"
        )
        return Image.open(frame_path).convert("RGB")

    def _load_boxes(self, video_id):
        anno_file = os.path.join(self.anno_dir, f"{video_id}.txt")
        with open(anno_file, 'r') as f:
            return [list(map(float, line.strip().split(','))) for line in f]

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        video_id, start_idx = self.sequence_indices[idx]
        boxes = self._load_boxes(video_id)
        
        # Load frames and get their original dimensions
        frames = []
        orig_dims = []
        for i in range(self.seq_length + 1):
            frame = self._load_frame(video_id, start_idx + i)
            orig_dims.append(frame.size)  # (width, height)
            frames.append(frame.resize((self.img_size, self.img_size)))
        
        # Process input frames and boxes
        input_frames = []
        input_boxes = []
        for i in range(self.seq_length):
            # Scale the bounding box coordinates
            orig_w, orig_h = orig_dims[i]
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            
            x, y, w, h = boxes[start_idx + i]
            scaled_box = [
                x * scale_x,
                y * scale_y,
                w * scale_x,
                h * scale_y
            ]
            
            input_frames.append(frames[i])
            input_boxes.append(scaled_box)
        
        # Scale target box
        orig_w, orig_h = orig_dims[-1]
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        x, y, w, h = boxes[start_idx + self.seq_length]
        target_box = [
            x * scale_x,
            y * scale_y,
            w * scale_x,
            h * scale_y
        ]
        
        # Apply transforms
        processed_frames = [self.transform(frame) for frame in input_frames]
        input_tensor = torch.stack(processed_frames)
        target_tensor = torch.tensor(target_box, dtype=torch.float32)
        
        return input_tensor, target_tensor

    def visualize_sample(self, idx, show_boxes=True):
        video_id, start_idx = self.sequence_indices[idx]
        boxes = self._load_boxes(video_id)
        
        # Load and resize frames
        frames = []
        orig_dims = []
        for i in range(self.seq_length):
            frame = self._load_frame(video_id, start_idx + i)
            orig_dims.append(frame.size)
            frames.append(frame.resize((self.img_size, self.img_size)))
        
        fig, axs = plt.subplots(1, self.seq_length, figsize=(20, 4))
        for i, frame in enumerate(frames):
            # Scale the bounding box
            orig_w, orig_h = orig_dims[i]
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            
            x, y, w, h = boxes[start_idx + i]
            scaled_box = [
                x * scale_x,
                y * scale_y,
                w * scale_x,
                h * scale_y
            ]
            
            axs[i].imshow(frame)
            if show_boxes:
                axs[i].add_patch(plt.Rectangle(
                    (scaled_box[0], scaled_box[1]),
                    scaled_box[2], scaled_box[3],
                    linewidth=2, edgecolor='r', facecolor='none'
                ))
            axs[i].set_title(f"Frame {i+1}\nBox: {[round(c,1) for c in scaled_box]}")
            axs[i].axis('off')
        
        plt.suptitle(f"Sample {idx} | Video: {video_id}", y=1.05)
        plt.tight_layout()
        plt.show()


def TrackingNetDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
