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
from sklearn.preprocessing import PowerTransformer
import numpy as np
import random

# Reproducibility
seed = 2024645714053328980 % (2**64)
random.seed(seed)
np.random.seed(seed & 0xFFFFFFFF)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Yeo-Johnson transformer
transformer = PowerTransformer(method='yeo-johnson')

# Dataset
class TubeDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.labels = self.data['conc'].values.reshape(-1, 1)
        self.transformed_labels = transformer.fit_transform(self.labels).flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.transformed_labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transforms
transform = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Load and split dataset
csv_path = '/home/noob/koty/new_before_last/sd_with_image_path.csv'
full_dataset = TubeDataset(csv_path=csv_path, transform=transform)
indices = list(range(len(full_dataset)))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# CNN model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),  # Add this
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(1)

model = ImprovedCNN()

# criterion = nn.SmoothL1Loss()
criterion = nn.HuberLoss(delta=1.0)
# criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

num_epochs = 250
scheduler = OneCycleLR(
    optimizer, max_lr=0.005,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)

# Evaluation function with inverse transform
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs)
            preds.extend(outputs.numpy())
            targets.extend(labels.numpy())
    preds = transformer.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    targets = transformer.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()
    loss = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return loss, r2

# Helper function to inverse transform single values
def inverse_single(val):
    return transformer.inverse_transform([[val]])[0][0]

# Function to get inverse-transformed predictions and labels
def get_preds_labels(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            preds.extend(outputs.numpy())
            labels.extend(targets.numpy())
    preds = transformer.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    labels = transformer.inverse_transform(np.array(labels).reshape(-1, 1)).flatten()
    return preds, labels

# Training loop with evaluation and scheduler step
best_val_loss = float('inf')

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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

torch.save(model.state_dict(), 'bb_model.pth')
torch.save(model.state_dict(), 'final_model_full.pth')

# Load best model checkpoint
model.load_state_dict(torch.load('bb_model.pth'))

import matplotlib.pyplot as plt

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

# Get predictions and true values for all sets (inverse transformed)
train_preds, train_labels = get_preds_labels(train_loader)
val_preds, val_labels = get_preds_labels(val_loader)
test_preds, test_labels = get_preds_labels(test_loader)

# Plot Predicted vs Actual
plt.figure(figsize=(7, 7))
plt.scatter(train_labels, train_preds, c='blue', alpha=0.6, label='Train', edgecolors='k')
plt.scatter(val_labels, val_preds, c='green', alpha=0.6, label='Validation', edgecolors='k')
plt.scatter(test_labels, test_preds, c='red', alpha=0.6, label='Test', edgecolors='k')

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

# Sample-wise predictions on test, val, and train with inverse transform applied
model.eval()
print("Sample-wise predictions on test set:")
with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        for i in range(len(labels)):
            y_true = inverse_single(labels[i].item())
            y_pred = inverse_single(outputs[i].item())
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

print("Sample-wise predictions on validation set:")
with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs)
        for i in range(len(labels)):
            y_true = inverse_single(labels[i].item())
            y_pred = inverse_single(outputs[i].item())
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")

print("Sample-wise predictions on train set:")
with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = model(imgs)
        for i in range(len(labels)):
            y_true = inverse_single(labels[i].item())
            y_pred = inverse_single(outputs[i].item())
            error = abs(y_true - y_pred)
            print(f"Y_actual: {y_true:.4f}, Y_predicted: {y_pred:.4f}, Error: {error:.4f}")
