#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 23:54:59 2025

@author: noob
"""

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

# Dataset class with log transformation
class TubeDataset(Dataset):
    def __init__(self, csv_path, transform=None, use_log_transform=True):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.use_log_transform = use_log_transform
        
        # Check for non-positive values if using log transform
        if self.use_log_transform:
            conc_values = self.data['conc'].values
            if np.any(conc_values <= 0):
                print(f"Warning: Found {np.sum(conc_values <= 0)} non-positive concentration values.")
                print(f"Min concentration: {np.min(conc_values)}")
                # Add small epsilon to avoid log(0) or log(negative)
                self.epsilon = 1e-8
                print(f"Adding epsilon={self.epsilon} to all concentration values for log transform.")
            else:
                self.epsilon = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        
        conc_value = float(self.data.iloc[idx]['conc'])
        
        if self.use_log_transform:
            # Apply log transformation to the target
            label = torch.tensor(np.log(conc_value + self.epsilon), dtype=torch.float32)
        else:
            label = torch.tensor(conc_value, dtype=torch.float32)
            
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
full_dataset = TubeDataset(csv_path='/home/noob/koty/new_before_last/project-ano/predictions-stage2/enhanced/sd_with_image_path.csv', 
                          transform=transform, use_log_transform=True)
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
class TinyConvRegressionModel(nn.Module):
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

# Loss function and optimizer
criterion = nn.L1Loss()
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
)

# Modified evaluation function with inverse log transform
def evaluate(model, loader, use_log_transform=True, epsilon=0):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            
            if use_log_transform:
                # Inverse log transform for predictions and targets
                preds_original = np.exp(outputs.numpy()) - epsilon
                targets_original = np.exp(labels.numpy()) - epsilon
            else:
                preds_original = outputs.numpy()
                targets_original = labels.numpy()
                
            preds.extend(preds_original)
            targets.extend(targets_original)
            
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

# Get epsilon value from dataset for inverse transform
epsilon = full_dataset.epsilon if hasattr(full_dataset, 'epsilon') else 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)  # Loss calculated on log-transformed values
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    
    # Evaluate with inverse transform to get original scale metrics
    train_eval_loss, train_r2 = evaluate(model, train_loader, use_log_transform=True, epsilon=epsilon)
    val_loss, val_r2 = evaluate(model, val_loader, use_log_transform=True, epsilon=epsilon)
    test_loss, test_r2 = evaluate(model, test_loader, use_log_transform=True, epsilon=epsilon)

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
        
torch.save(model.state_dict(), 'bb_model.pth')
torch.save(model.state_dict(), 'final_model_full.pth')

# Load best model checkpoint
model.load_state_dict(torch.load('bb_model.pth'))

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss (Original Scale)')
plt.title('Loss over Epochs (After Inverse Log Transform)')
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
plt.title('R² over Epochs (Original Scale)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("r2_plot.png")
plt.show()

# Modified function to get predictions with inverse transform
def get_preds_labels(loader, use_log_transform=True, epsilon=0):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            
            if use_log_transform:
                # Inverse log transform
                preds_original = np.exp(outputs.numpy()) - epsilon
                targets_original = np.exp(targets.numpy()) - epsilon
            else:
                preds_original = outputs.numpy()
                targets_original = targets.numpy()
                
            preds.extend(preds_original)
            labels.extend(targets_original)
            
    return np.array(preds), np.array(labels)

# Get predictions and true values for all sets (in original scale)
train_preds, train_labels = get_preds_labels(train_loader, use_log_transform=True, epsilon=epsilon)
val_preds, val_labels = get_preds_labels(val_loader, use_log_transform=True, epsilon=epsilon)
test_preds, test_labels = get_preds_labels(test_loader, use_log_transform=True, epsilon=epsilon)

# Plot Predicted vs Actual (Original Scale)
plt.figure(figsize=(7, 7))
plt.scatter(train_labels, train_preds, c='blue', alpha=0.6, label='Train', edgecolors='k')
plt.scatter(val_labels, val_preds, c='green', alpha=0.6, label='Validation', edgecolors='k')
plt.scatter(test_labels, test_preds, c='red', alpha=0.6, label='Test', edgecolors='k')

# Identity line (perfect prediction)
min_val = min(train_labels.min(), val_labels.min(), test_labels.min())
max_val = max(train_labels.max(), val_labels.max(), test_labels.max())
plt.plot([min_val, max_val], [min_val, max_val], 'black', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Concentration (Original Scale)')
plt.ylabel('Predicted Concentration (Original Scale)')
plt.title('Predicted vs Actual Concentration (All Sets) - Original Scale')
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
plt.xlabel('Actual Concentration (Original Scale)')
plt.ylabel('Residual (Predicted - Actual)')
plt.title('Residual Plot: Predicted vs Actual Error (Original Scale)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()

# Sample-wise predictions with inverse transform
print("Sample-wise predictions on test set (Original Scale):")
model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        # Inverse transform both predictions and labels
        outputs_original = torch.exp(outputs) - epsilon
        labels_original = torch.exp(labels) - epsilon
        
        for i in range(len(labels)):
            y_true = labels_original[i].item()
            y_pred = outputs_original[i].item()
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

    print("\nSample-wise predictions on validation set (Original Scale):")
    for imgs, labels in val_loader:
        outputs = model(imgs)
        outputs_original = torch.exp(outputs) - epsilon
        labels_original = torch.exp(labels) - epsilon
        
        for i in range(len(labels)):
            y_true = labels_original[i].item()
            y_pred = outputs_original[i].item()
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

    print("\nSample-wise predictions on training set (Original Scale):")
    for imgs, labels in train_loader:
        outputs = model(imgs)
        outputs_original = torch.exp(outputs) - epsilon
        labels_original = torch.exp(labels) - epsilon
        
        for i in range(len(labels)):
            y_true = labels_original[i].item()
            y_pred = outputs_original[i].item()
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

# Function to make predictions on new data (with inverse transform)
def predict_concentration(model, image_path, transform, epsilon=0):
    """
    Make a prediction on a single image and return the concentration in original scale.
    """
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        log_prediction = model(image_tensor)
        # Inverse log transform
        original_prediction = torch.exp(log_prediction) - epsilon
        
    return original_prediction.item()

print(f"\nLog transformation applied with epsilon={epsilon}")
print("All predictions and evaluations are now shown in the original concentration scale.")