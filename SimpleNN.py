import torch
import torch.nn as nn
import numpy as np
from Util import load_images

import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)

from experiment_log import log_experiment
import datetime

class SimpleCNN(nn.Module):
  def __init__(self, input_channels, image_dimensions):
    super(SimpleCNN, self).__init__()
    self.image_dimensions = image_dimensions
    # in_channels=1 because our image is grayscale (if color images, then in_channels=3 for RGB).
    # out_channels=16 means we have 16 filters, each filter of size 3x3x1.
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
    
    # in_channels=16 because our out_channels=16 from previous layer.
    # out_channels=32 means we are using 32 filters, each filter of size 3x3x16,
    # in this layer.
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    
    # Max Pooling Layer: downsample by a factor of 2.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Fully Connected Layer 1: input size = 7 * 7 * 32 (from feature maps), output size = 128.
    #self.fc1 = nn.Linear(in_features= 7 * 7 * 32, out_features=128)
    self.fc1 = nn.Linear(in_features= self.conv2.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4), out_features=128)
    
    # Fully Connected Layer 2: input size = 128, output size = 10 (for 10 output classes).
    self.fc2 = nn.Linear(in_features=128, out_features=4)

    # Activation function
    self.relu = nn.ReLU()

  def forward(self, x):
    #print(f"x.shape={x.shape}\n")

    # Apply convolution + ReLU + pooling
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)

    # Flatten the feature maps 
    #x = x.view(-1, 7 * 7 * 32)
    x = x.view(-1, self.conv2.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4))


    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x

def train(model, criterion, optimizer, filepaths, labels, device, epochs, image_dimensions):
  # our hyper-parameters for training
  n_epochs = epochs
  batch_size = 16

  for epoch in range(n_epochs):
    # For tracking and printing our training-progress
    samples_trained = 0
    run_loss = 0
    correct_preds = 0
    total_samples = len(filepaths) 

    permutation = torch.randperm(total_samples)
    for i in range(0, total_samples, batch_size):
      indices = permutation[i : i+batch_size]
      #batch_inputs = load_images(filepaths[indices], device)
      batch_inputs = load_images(filepaths[indices], device, dimensions=image_dimensions)
      
      #batch_labels = labels[indices]
      dup_factor   = batch_inputs.size(0) // len(indices)     # e.g. 3 if aug ×3
      batch_labels = labels[indices].repeat_interleave(dup_factor).to(device)
      
      # Forward pass: compute predicted outputs
      outputs = model(batch_inputs)

      # Compute loss
      loss = criterion(outputs, batch_labels)
      run_loss += loss.item()

      # Backward pass and optimization step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # Get probability-distributions
      probs = torch.softmax(outputs, dim=1)
      _, preds = torch.max(probs, dim=1)

      # Calculate some stats
      # samples_trained += len(indices)
      samples_trained += len(batch_labels)
      avg_loss = run_loss / samples_trained

      correct_preds += torch.sum(preds == batch_labels) # compare predictions with labels
      accuracy = correct_preds / float(samples_trained) # cast to float to get "accuracy" in decimal 

      print(f"Epoch {epoch+1} " +
            f"({samples_trained}/{total_samples}): " +
            f"Loss={avg_loss:.5f}, Accuracy={accuracy:.5f}")
      
def test(model, filepaths, labels, device, image_dimensions,
         train_batch, train_inputs, epochs, save_cm=False, model_notes=""):
  batch_size = 12
  samples_tested = 0
  correct_preds = 0
  total_samples = len(filepaths)

  all_preds = []
  all_targets = []

  
  filepaths = np.array(filepaths)
  labels = labels.clone()
  permutation = torch.randperm(total_samples)
  filepaths = filepaths[permutation.numpy()]
  labels = labels[permutation]
  
  ts = datetime.datetime.now().isoformat(timespec="seconds")

  model.eval()         
  with torch.no_grad(): 

    for i in range(0, total_samples, batch_size):
      batch_inputs = load_images(filepaths[i : i + batch_size], device, dimensions=image_dimensions)
      batch_labels = labels[i : i + batch_size]

      # Forward pass: compute predicted outputs
      outputs = model(batch_inputs)

      # Get probability-distributions
      probs = torch.softmax(outputs, dim=1)
      _, preds = torch.max(probs, dim=1)

      # Determine accuracy
      samples_tested += len(batch_labels)
      correct_preds += torch.sum(preds == batch_labels)
      accuracy = correct_preds / float(samples_tested)


      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(batch_labels.cpu().numpy())


      # Accuracy score
      print(f"({samples_tested}/{total_samples}): Accuracy={accuracy:.5f}")

      # Other metrics
      precision = precision_score(all_targets, all_preds, average="macro")
      recall    = recall_score   (all_targets, all_preds, average="macro")
      f1        = f1_score       (all_targets, all_preds, average="macro")

      print("\n=== Macro-averaged metrics on test set ===")
      print(f"Precision : {precision:.4f}")
      print(f"Recall    : {recall   :.4f}")
      print(f"F1-score  : {f1       :.4f}")

      # Confusion matrix plot
      cm = confusion_matrix(all_targets, all_preds)
      fig, ax = plt.subplots(figsize=(5, 5))
      disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=["apple",
                                                    "banana",
                                                    "mixed",
                                                    "orange"])
      disp.plot(ax=ax, values_format="d")
      ax.set_title("Confusion Matrix")
      plt.tight_layout()

      if save_cm:
         fname = f"confusion_matrix_{ts}.png".replace(":", "-")
         plt.savefig(fname, dpi=150)
         print(f"Confusion-matrix saved as {fname}")
      else:
         plt.show()

      log_experiment(model           = model,
                augmentation_tag= "test",      # or the augment flag you passed
                batch_size      = train_batch,  # 60 in your current test()
                total_inputs    = train_inputs,
                epochs          = epochs,
                accuracy        = accuracy,
                precision       = precision,
                recall          = recall,
                f1              = f1,
                confusion_matrix= cm,
                ts=ts)
      