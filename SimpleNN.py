import torch
import torch.nn as nn
import numpy as np
from Util import load_images

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    # in_channels=1 because our image is grayscale (if color images, then in_channels=3 for RGB).
    # out_channels=16 means we have 16 filters, each filter of size 3x3x1.
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # in_channels=16 because our out_channels=16 from previous layer.
    # out_channels=32 means we are using 32 filters, each filter of size 3x3x16,
    # in this layer.
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    
    # Max Pooling Layer: downsample by a factor of 2.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Fully Connected Layer 1: input size = 7 * 7 * 32 (from feature maps), output size = 128.
    self.fc1 = nn.Linear(in_features= 7 * 7 * 32, out_features=128)
    
    # Fully Connected Layer 2: input size = 128, output size = 10 (for 10 output classes).
    self.fc2 = nn.Linear(in_features=128, out_features=10)

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
    x = x.view(-1, 7 * 7 * 32)

    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x

def train(model, criterion, optimizer, filepaths, labels, device):
  # our hyper-parameters for training
  n_epochs = 6
  batch_size = 12 

  for epoch in range(n_epochs):
    # For tracking and printing our training-progress
    samples_trained = 0
    run_loss = 0
    correct_preds = 0
    total_samples = len(filepaths) 

    permutation = torch.randperm(total_samples)
    for i in range(0, total_samples, batch_size):
      indices = permutation[i : i+batch_size]
      batch_inputs = load_images(filepaths[indices])
      batch_labels = labels[indices]

      # Forward pass: coyympute predicted outputs
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
      
def test(model, filepaths, labels):
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
  


  for i in range(0, total_samples, batch_size):
    batch_inputs = load_images(filepaths[i : i + batch_size])
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



    print(f"({samples_tested}/{total_samples}): Accuracy={accuracy:.5f}")