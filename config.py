# Model Configuration
MODEL_CONFIG = {
    'input_size': 128,          # Input image size (assumed to be square)
    'hidden_size': 256,         # LSTM hidden size
    'num_layers': 2,            # Number of LSTM layers
    'bidirectional': True       # Whether to use bidirectional LSTM
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 32,           # Training batch size
    'num_epochs': 5,           # Number of training epochs
    'learning_rate': 0.001,     # Initial learning rate
    'patience': 3,              # Patience for LR reduction
    'lr_factor': 0.5,           # Factor for LR reduction
    'train_size': 9000,         # Number of training samples
    'val_size': 100,            # Number of validation samples
    'save_path': 'model.pth'    # Path to save trained model
}

# Dataset Configuration
DATA_CONFIG = {
    'img_size': 128,            # Size to resize images to (height, width)
    'seq_length': 5,            # Number of frames in input sequence
    'normalize_mean': [0.485, 0.456, 0.406],  # Normalization mean
    'normalize_std': [0.229, 0.224, 0.225],   # Normalization std
    'root_dir': './data'        # Root directory for dataset
}

# Image Processing Configuration
IMG_CONFIG = {
    'resize_x': 128,            # Width to resize images to
    'resize_y': 128,            # Height to resize images to
    'input_channels': 3         # Number of input channels (RGB)
}

# Loss Configuration
LOSS_CONFIG = {
    'type': 'L1Loss'            # Loss function to use
}

# You can access these configurations like:
# from config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG
