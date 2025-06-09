#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 23:07:21 2025
@author: noob
Hybrid CNN-XGBoost Model
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import random

# Set seed for reproducibility
seed = 13977357109135510523
seed = seed % (2**64)

random.seed(seed)
np.random.seed(seed & 0xFFFFFFFF)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Global seed set to: {seed}")

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

# Modified CNN Feature Extractor (no fully connected layers)
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)

# Hybrid CNN-XGBoost Model
class HybridCNNXGBoost:
    def __init__(self, cnn_epochs=50, xgb_params=None):
        self.cnn_feature_extractor = CNNFeatureExtractor()
        self.cnn_epochs = cnn_epochs
        self.xgb_model = None
        
        # Default XGBoost parameters
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 10,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 1
            }
        else:
            self.xgb_params = xgb_params
    
    def extract_features(self, dataloader):
        """Extract features using the CNN feature extractor"""
        self.cnn_feature_extractor.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                features = self.cnn_feature_extractor(images)
                features_list.append(features.numpy())
                labels_list.append(labels.numpy())
        
        return np.vstack(features_list), np.concatenate(labels_list)
    
    def train_cnn_features(self, train_loader, val_loader, lr=0.001):
        """Train the CNN feature extractor"""
        print("Training CNN feature extractor...")
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.cnn_feature_extractor.parameters(), lr=lr, weight_decay=1e-3)
        
        # Temporary regressor head for CNN training
        temp_regressor = nn.Sequential(
            nn.Linear(32 * 10 * 10, 16),  # Adjust based on your feature size
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Full temporary model
        temp_model = nn.Sequential(self.cnn_feature_extractor, temp_regressor)
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=lr, weight_decay=1e-3)
        
        for epoch in range(self.cnn_epochs):
            temp_model.train()
            running_loss = 0.0
            
            for images, targets in train_loader:
                temp_optimizer.zero_grad()
                outputs = temp_model(images).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                temp_optimizer.step()
                running_loss += loss.item()
            
            if epoch % 10 == 0:
                temp_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, targets in val_loader:
                        outputs = temp_model(images).squeeze()
                        val_loss += criterion(outputs, targets).item()
                
                print(f"CNN Epoch {epoch}: Train Loss: {running_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}")
        
        print("CNN feature extractor training completed!")
    
    def fit(self, train_loader, val_loader):
        """Train the hybrid model"""
        # Step 1: Train CNN feature extractor
        self.train_cnn_features(train_loader, val_loader)
        
        # Step 2: Extract features for XGBoost
        print("Extracting features for XGBoost...")
        train_features, train_labels = self.extract_features(train_loader)
        val_features, val_labels = self.extract_features(val_loader)
        
        print(f"Train features shape: {train_features.shape}")
        print(f"Val features shape: {val_features.shape}")
        
        # Step 3: Train XGBoost
        print("Training XGBoost model...")
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        
        self.xgb_model.fit(
            train_features, train_labels,
            eval_set=[(train_features, train_labels), (val_features, val_labels)],
            verbose=True
        )
        
        print("Hybrid model training completed!")
    
    def predict(self, dataloader):
        """Make predictions using the hybrid model"""
        features, true_labels = self.extract_features(dataloader)
        predictions = self.xgb_model.predict(features)
        return predictions, true_labels
    
    def evaluate(self, dataloader):
        """Evaluate the hybrid model"""
        predictions, true_labels = self.predict(dataloader)
        
        mae = mean_absolute_error(true_labels, predictions)
        mse = mean_squared_error(true_labels, predictions)
        r2 = r2_score(true_labels, predictions)
        mape = mean_absolute_percentage_error(true_labels, predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'true_labels': true_labels
        }

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

# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': seed,
    'n_jobs': -1,
    'verbosity': 1
}

# Create and train hybrid model
hybrid_model = HybridCNNXGBoost(cnn_epochs=2000, xgb_params=xgb_params)
hybrid_model.fit(train_loader, val_loader)

# Evaluate on all splits
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

train_results = hybrid_model.evaluate(train_loader)
val_results = hybrid_model.evaluate(val_loader)  
test_results = hybrid_model.evaluate(test_loader)

def print_results(split_name, results):
    print(f"\n{split_name} Results:")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  R²: {results['r2']:.4f}")
    print(f"  MAPE: {results['mape']*100:.2f}%")

print_results("Train", train_results)
print_results("Validation", val_results)
print_results("Test", test_results)

# Detailed predictions
def show_detailed_predictions(split_name, results):
    print(f"\n{split_name} Set Detailed Predictions:")
    predictions = results['predictions']
    true_labels = results['true_labels']
    
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        error = abs(pred - true)
        print(f"Sample {i+1}: Y_actual: {true:.4f}, Y_predicted: {pred:.4f}, Error: {error:.4f}")

show_detailed_predictions("Test", test_results)
show_detailed_predictions("Validation", val_results)
show_detailed_predictions("Train", train_results)

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Predicted vs Actual
plt.subplot(1, 3, 1)
plt.scatter(train_results['true_labels'], train_results['predictions'], 
           c='blue', alpha=0.6, label='Train', edgecolors='k')
plt.scatter(val_results['true_labels'], val_results['predictions'], 
           c='green', alpha=0.6, label='Validation', edgecolors='k')
plt.scatter(test_results['true_labels'], test_results['predictions'], 
           c='red', alpha=0.6, label='Test', edgecolors='k')

# Identity line
all_true = np.concatenate([train_results['true_labels'], 
                          val_results['true_labels'], 
                          test_results['true_labels']])
min_val, max_val = all_true.min(), all_true.max()
plt.plot([min_val, max_val], [min_val, max_val], 'black', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Concentration')
plt.ylabel('Predicted Concentration')
plt.title('Hybrid CNN-XGBoost: Predicted vs Actual')
plt.legend()
plt.grid(True)

# Plot 2: Residuals
plt.subplot(1, 3, 2)
train_residuals = train_results['predictions'] - train_results['true_labels']
val_residuals = val_results['predictions'] - val_results['true_labels']
test_residuals = test_results['predictions'] - test_results['true_labels']

plt.scatter(train_results['true_labels'], train_residuals, alpha=0.6, label='Train', edgecolors='k', c='blue')
plt.scatter(val_results['true_labels'], val_residuals, alpha=0.6, label='Validation', edgecolors='k', c='green')
plt.scatter(test_results['true_labels'], test_residuals, alpha=0.6, label='Test', edgecolors='k', c='red')

plt.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Zero Error Line')
plt.xlabel('Actual Concentration')
plt.ylabel('Residual (Predicted - Actual)')
plt.title('Hybrid CNN-XGBoost: Residual Plot')
plt.legend()
plt.grid(True)

# Plot 3: Feature Importance (if available)
plt.subplot(1, 3, 3)
if hasattr(hybrid_model.xgb_model, 'feature_importances_'):
    importance = hybrid_model.xgb_model.feature_importances_
    feature_indices = range(len(importance))
    
    # Show top 20 most important features
    top_indices = np.argsort(importance)[-20:]
    top_importance = importance[top_indices]
    
    plt.barh(range(len(top_importance)), top_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Index')
    plt.title('XGBoost Feature Importance (Top 20)')
    plt.yticks(range(len(top_importance)), [f'F{i}' for i in top_indices])
    plt.grid(True)

plt.tight_layout()
plt.savefig("hybrid_cnn_xgb_results.png", dpi=300, bbox_inches='tight')
plt.show()

# Save models
torch.save(hybrid_model.cnn_feature_extractor.state_dict(), 'cnn_feature_extractor.pth')
hybrid_model.xgb_model.save_model('xgb_regressor.json')

print(f"\nModels saved:")
print(f"- CNN Feature Extractor: cnn_feature_extractor.pth")
print(f"- XGBoost Regressor: xgb_regressor.json")

# Comparison with original approach
print(f"\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"Hybrid CNN-XGBoost Test R²: {test_results['r2']:.4f}")
print(f"Hybrid CNN-XGBoost Test MAE: {test_results['mae']:.4f}")
print("This hybrid approach combines the spatial feature learning of CNNs")
print("with the powerful gradient boosting capabilities of XGBoost.")