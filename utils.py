import torch
import matplotlib.pyplot as plt

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def plot_frame_with_bbox(frame, box, title="Frame with BBox"):
    fig, ax = plt.subplots(1)
    ax.imshow(frame)
    rect = plt.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title(title)
    plt.show()