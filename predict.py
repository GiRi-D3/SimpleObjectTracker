import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def box_to_rect(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def plot_boxes(frame, true_box, pred_box, counter):
    true_rect = box_to_rect(true_box)
    pred_rect = box_to_rect(pred_box)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(frame)
    
    # Ground truth box (green)
    plt.plot([true_rect[0], true_rect[2]], [true_rect[1], true_rect[1]], 'g-', linewidth=2)
    plt.plot([true_rect[0], true_rect[2]], [true_rect[3], true_rect[3]], 'g-', linewidth=2)
    plt.plot([true_rect[0], true_rect[0]], [true_rect[1], true_rect[3]], 'g-', linewidth=2)
    plt.plot([true_rect[2], true_rect[2]], [true_rect[1], true_rect[3]], 'g-', linewidth=2)
    
    # Predicted box (red)
    plt.plot([pred_rect[0], pred_rect[2]], [pred_rect[1], pred_rect[1]], 'r-', linewidth=2)
    plt.plot([pred_rect[0], pred_rect[2]], [pred_rect[3], pred_rect[3]], 'r-', linewidth=2)
    plt.plot([pred_rect[0], pred_rect[0]], [pred_rect[1], pred_rect[3]], 'r-', linewidth=2)
    plt.plot([pred_rect[2], pred_rect[2]], [pred_rect[1], pred_rect[3]], 'r-', linewidth=2)
    
    plt.title(f"Sample {counter+1} — True (green) vs Predicted (red)")
    plt.axis('off')
    plt.show()

def test_SimpleObjectTracker(dataset, model, denormalize, device=None, num_samples=50):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    model = model.to(device)
    model.eval()
    
    counter = 0
    with torch.no_grad():
        for inputs, true_box in loader:
            if counter >= num_samples:
                break
            
            inputs = inputs.to(device)
            pred_box = model(inputs).cpu().numpy()[0]
            true_box = true_box.numpy()[0]
            
            last_frame = inputs[0, -1].cpu()
            last_frame = denormalize(last_frame).permute(1, 2, 0).numpy()
            
            plot_boxes(last_frame, true_box, pred_box, counter)
            counter += 1
