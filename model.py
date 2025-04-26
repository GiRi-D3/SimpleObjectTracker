import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleObjectTracker(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super(SimpleObjectTracker, self).__init__()
        
        # Enhanced CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Final feature map: 256x4x4
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256*4*4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Regression head with sigmoid for normalized coordinates
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Output normalized coordinates [0,1]
        )
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process all frames in parallel using reshape
        x = x.view(-1, *x.shape[2:])  # (batch*seq_len, 3, H, W)
        features = self.encoder(x)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, 256*4*4)
        
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]  # Take last timestep
        
        # Predict normalized bounding box (x,y,w,h) in [0,1] range
        bbox = self.regressor(last_out) * 128  # Scale to 128x128 image size
        
        return bbox