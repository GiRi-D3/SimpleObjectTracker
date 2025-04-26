import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def train_SimpleObjectTracker(model, dataset, batch_size=32, num_epochs=5, learning_rate=0.001, save_path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set train and validation sizes
    train_size = 9000
    val_size = 100
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training', leave=False)

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            avg_train_loss = train_loss / ((loop.n + 1) * batch_size)
            loop.set_postfix(avg_loss=avg_train_loss)

        model.eval()
        val_loss = 0.0
        loop = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Validating', leave=False)

        with torch.no_grad():
            for inputs, targets in loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets).item()
                val_loss += batch_loss * inputs.size(0)
                avg_val_loss = val_loss / ((loop.n + 1) * batch_size)
                loop.set_postfix(avg_loss=avg_val_loss)

        scheduler.step(val_loss)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), save_path)
