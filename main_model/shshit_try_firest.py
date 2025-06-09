#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 23:07:21 2025
@author: noob
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

import random



seed = 2024645714053328980
seed = seed % (2**64)  # Ensure seed fits in 64-bit integer

random.seed(seed)                 # Python random module
np.random.seed(seed & 0xFFFFFFFF)  # NumPy (32-bit seed)
torch.manual_seed(seed)           # PyTorch CPU
torch.cuda.manual_seed(seed)      # PyTorch CUDA (if available)
torch.cuda.manual_seed_all(seed)  # All GPUs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {seed}")


current_seed = torch.initial_seed()
print(f"Current PyTorch CPU RNG seed: {current_seed}")

with open("seed_log.txt", "a") as f:
    f.write(f"Current PyTorch CPU RNG seed: {current_seed}\n")

# Dataset class
class TubeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(float(self.data.iloc[idx]['conc']), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Dataset loading and split
full_dataset = TubeDataset(csv_path='/home/noob/koty/new_before_last/project-ano/predictions-stage2/enhanced/sd_with_image_path.csv', transform=transform)
indices = list(range(len(full_dataset)))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Tiny ConvNet Model
# class TinyConvRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(4),  # Added here
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.regressor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(4 * 10 * 10, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.regressor(x)
#         return x.squeeze(1)
import torch

class TinyConvRegressionModel(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 4, kernel_size=3, padding=1),
    #         nn.BatchNorm2d(4),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2),
    #     )
    #     # Placeholder for regressor; will create after computing flattened size
    #     self.regressor = None
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.regressor = None


    def forward(self, x):
        x = self.features(x)
        if self.regressor is None:
            flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ).to(x.device)
        x = self.regressor(x)
        return x.squeeze(1)


model = TinyConvRegressionModel()



class HybridLoss(nn.Module):
    """
    Combination of absolute and relative loss
    Formula: alpha * MAE + (1-alpha) * RelativeMAE
    """
    def __init__(self, alpha=0.5, epsilon=1e-6):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        absolute_loss = self.mae(pred, target)
        relative_loss = torch.mean(torch.abs(pred - target) / (torch.abs(target) + self.epsilon))
        return self.alpha * absolute_loss + (1 - self.alpha) * relative_loss    
    
    
criterion = nn.L1Loss()
# criterion = HybridLoss(alpha=0.3, epsilon=1e-6)  # 30% absolute, 70% relative
optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-3)

# OneCycleLR Scheduler
num_epochs = 300
steps_per_epoch = len(train_loader)

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.005,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0,
    # verbose=True
)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            preds.extend(outputs.numpy())
            targets.extend(labels.numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    loss = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return loss, r2

# Training and Tracking
best_val_loss = float('inf')
patience = 15
counter = 0

train_losses, val_losses, test_losses = [], [], []
train_r2s, val_r2s, test_r2s = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    train_eval_loss, train_r2 = evaluate(model, train_loader)
    val_loss, val_r2 = evaluate(model, val_loader)
    test_loss, test_r2 = evaluate(model, test_loader)

    # Append to tracking lists
    train_losses.append(train_eval_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    train_r2s.append(train_r2)
    val_r2s.append(val_r2)
    test_r2s.append(test_r2)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_eval_loss:.4f}, R²: {train_r2:.4f} | "
          f"Val Loss: {val_loss:.4f}, R²: {val_r2:.4f} | "
          f"Test Loss: {test_loss:.4f}, R²: {test_r2:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.20f}")

    # Early stopping (optional)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        pass
        # counter += 1
        # if counter >= patience:
        #     print(f"Early stopping triggered at epoch {epoch+1}")
        #     break
        
        
torch.save(model.state_dict(), 'bb_model.pth')

# torch.save(model, 'final_model_full.pth')    
torch.save(model.state_dict(), 'final_model_full.pth')

# Load best model checkpoint
model.load_state_dict(torch.load('bb_model.pth'))

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# Plot R² Score
plt.figure(figsize=(10, 5))
plt.plot(train_r2s, label='Train R²')
plt.plot(val_r2s, label='Validation R²')
plt.plot(test_r2s, label='Test R²')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('R² over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("r2_plot.png")
plt.show()




import matplotlib.pyplot as plt

def get_preds_labels(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            preds.extend(outputs.numpy())
            labels.extend(targets.numpy())
    return np.array(preds), np.array(labels)

# Get predictions and true values for all sets
train_preds, train_labels = get_preds_labels(train_loader)
val_preds, val_labels = get_preds_labels(val_loader)
test_preds, test_labels = get_preds_labels(test_loader)

# Plot
plt.figure(figsize=(7, 7))
plt.scatter(train_labels, train_preds, c='blue', alpha=0.6, label='Train', edgecolors='k')
plt.scatter(val_labels, val_preds, c='green', alpha=0.6, label='Validation', edgecolors='k')
plt.scatter(test_labels, test_preds, c='red', alpha=0.6, label='Test', edgecolors='k')

# Identity line (perfect prediction)
min_val = min(train_labels.min(), val_labels.min(), test_labels.min())
max_val = max(train_labels.max(), val_labels.max(), test_labels.max())
plt.plot([min_val, max_val], [min_val, max_val], 'black', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Concentration')
plt.ylabel('Predicted Concentration')
plt.title('Predicted vs Actual Concentration (All Sets)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Residual Plot
def plot_residuals(preds, labels, dataset_name, color):
    residuals = preds - labels
    plt.scatter(labels, residuals, alpha=0.6, label=f'{dataset_name}', edgecolors='k', c=color)

plt.figure(figsize=(10, 6))
plot_residuals(train_preds, train_labels, 'Train', 'blue')
plot_residuals(val_preds, val_labels, 'Validation', 'green')
plot_residuals(test_preds, test_labels, 'Test', 'red')

plt.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error Line')
plt.xlabel('Actual Concentration')
plt.ylabel('Residual (Predicted - Actual)')
plt.title('Residual Plot: Predicted vs Actual Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()



model.eval()
print("Sample-wise predictions on test set:")
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        for i in range(len(labels)):
            x = imgs[i]  # image tensor (3x20x20)
            y_true = labels[i].item()
            y_pred = outputs[i].item()
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

    for imgs, labels in val_loader:
        outputs = model(imgs)
        for i in range(len(labels)):
            x = imgs[i]  # image tensor (3x20x20)
            y_true = labels[i].item()
            y_pred = outputs[i].item()
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

    for imgs, labels in train_loader:
        outputs = model(imgs)
        for i in range(len(labels)):
            x = imgs[i]  # image tensor (3x20x20)
            y_true = labels[i].item()
            y_pred = outputs[i].item()
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")





# def show_before_and_after(csv_path, transform, num_images=5):
#     import matplotlib.pyplot as plt
#     from PIL import Image
#     import torchvision.transforms.functional as F

#     df = pd.read_csv(csv_path)

#     plt.figure(figsize=(10, 4 * num_images))

#     for i in range(num_images):
#         row = df.iloc[i]
#         img_path = row['image_path']
#         label = row['Conc']
        
#         # Load original image
#         original_img = Image.open(img_path).convert('RGB')

#         # Transformed image
#         transformed_img_tensor = transform(original_img)
#         transformed_img_np = transformed_img_tensor.numpy().transpose((1, 2, 0))
#         transformed_img_np = transformed_img_np * 0.5 + 0.5  # De-normalize [0,1]

#         # Plot original
#         plt.subplot(num_images, 2, 2 * i + 1)
#         plt.imshow(original_img)
#         plt.title(f"Original Image\nConc: {label:.2f}")
#         plt.axis('off')

#         # Plot transformed
#         plt.subplot(num_images, 2, 2 * i + 2)
#         plt.imshow(transformed_img_np)
#         plt.title(f"Transformed Image\n(Resized + Normalized)")
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()



# show_before_and_after(csv_path='try.csv', transform=transform, num_images=5)