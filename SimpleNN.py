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
import os

class SimpleCNN(nn.Module):
  def __init__(self, input_channels, image_dimensions):
    super(SimpleCNN, self).__init__()
    self.image_dimensions = image_dimensions
    # in_channels = no. of input channels
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
    # in_channels=16 because our out_channels=16 from previous layer.
    # out_channels=32 means we are using 32 filters, each filter of size 3x3x16,
    # in this layer.
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

    
    # Max Pooling Layer: downsample by a factor of 2.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # Fully Connected Layer 1: input size = 7 * 7 * 32 (from feature maps), output size = 128.
    self.fc1 = nn.Linear(in_features= self.conv3.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4), out_features=128)
    # self.fc1 = nn.Linear(in_features= self.conv2.out_channels * (image_dimensions[0]/4) * (image_dimensions[1]/4), out_features=128)
    
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

    x = self.conv3(x)
    x = self.relu(x)
    x = self.pool(x)

    # Flatten the feature maps 
    x = x.view(-1, self.conv3.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4))

    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x

def train(model, criterion, optimizer, filepaths, labels, device, epochs, n_batch_size, image_dimensions, augment):
    # >>> INSERT: init lists & early‐stop vars for plotting
  train_losses = []
  train_accs   = []
  best_train_loss   = float('inf')
  patience = 3   # adjust if you like
  stop_training = False

    # >>> SPLIT OUT A VALIDATION SET
  from sklearn.model_selection import train_test_split
  train_paths, val_paths, train_labels, val_labels = train_test_split(
      filepaths, labels.cpu().numpy(),
      test_size=0.1, stratify=labels.cpu().numpy(), random_state=42
  )
  train_labels = torch.tensor(train_labels, device=device)
  val_labels   = torch.tensor(val_labels,   device=device)

  val_losses = []
  val_accs   = []
  best_val_loss = float('inf')
  
  # our hyper-parameters for training
  n_epochs = epochs 
  batch_size = n_batch_size # processes 16 images at one go
  total_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

  for epoch in range(n_epochs):
    model.train()
    # For tracking and printing our training-progress
    samples_trained = 0
    run_loss = 0
    correct_preds = 0
    total_samples = len(train_paths)
    
    if stop_training == True:
      break
    # randomly selects images 
    #permutation = torch.randperm(total_samples)
      # shuffle only the train set
    permutation = torch.randperm(len(train_paths))

    for i in range(0, total_samples, batch_size):
      indices = permutation[i : i+batch_size]
      #batch_inputs = load_images(filepaths[indices], device, dimensions=image_dimensions, augmentation=augment)
      batch_inputs = load_images(train_paths[indices], device, dimensions=image_dimensions, augmentation=augment)

      # batch_labels = labels[indices]
      dup_factor   = batch_inputs.size(0) // len(indices)     # e.g. 3 if aug ×3
      batch_labels = train_labels[indices].repeat_interleave(dup_factor, dim=0).to(device)
      
      
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

      # >>> INSERT: end-of-epoch metrics & early stopping on train loss
      # since this print is inside the batch-loop, we only want to record
      # when we've seen all samples in this epoch:
      # epoch_loss = 0
      # best_train_loss = 0
      if samples_trained == total_samples:
        epoch_loss = run_loss / total_samples
        epoch_acc  = correct_preds.float() / total_samples


        train_losses.append(avg_loss)           # avg_loss is already a float
        train_accs.append(accuracy.item())      # <- extract a Python float

                # >>> INSERT: validation pass
        # >>> CORRECTED VALIDATION PASS on val_paths/val_labels
        model.eval()

        v_loss, v_corr = 0.0, 0
        val_count = len(val_paths)
        # no need to shuffle for validation
        for start in range(0, val_count, batch_size):
            end = start + batch_size
            v_batch = val_paths[start:end]
            v_imgs  = load_images(v_batch, device,
                                  dimensions=image_dimensions,
                                  augmentation="224")
            v_lbls  = val_labels[start:end]
            v_out   = model(v_imgs)
            l       = criterion(v_out, v_lbls)
            v_loss += l.item() * v_lbls.size(0)
            v_corr += (v_out.argmax(1) == v_lbls).sum().item()
        epoch_val_loss = v_loss / val_count
        epoch_val_acc  = v_corr / val_count
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        print(f"Validation loss: {epoch_val_loss:.5f}, " +
              f"Validation accuracy: {epoch_val_acc:.5f}")

        model.train()

          # early stop on training loss

        if epoch_loss < best_train_loss:
          best_train_loss = epoch_loss

        if epoch_val_loss < best_val_loss:
          best_val_loss = epoch_val_loss
          epochs_no_improve = 0
          torch.save(model.state_dict(), 'best_model.pt')
        else:
          epochs_no_improve += 1
          if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            stop_training = True

        print(f"Epoch loss: {epoch_val_loss} " +
              f"Best loss: {best_val_loss} " +
          f"Epochs no improve: {epochs_no_improve} " +
              f"Stop training: {stop_training}")
        

        # # live‐plot train curves
        # plt.ion()
        # plt.clf()
        # # Loss
        # plt.subplot(1,2,1)
        # plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        # plt.title('Loss vs Epoch')
        # plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        # # Acc
        # plt.subplot(1,2,2)
        # plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
        # plt.title('Accuracy vs Epoch')
        # plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        # plt.pause(0.1)


      print(f"Epoch {epoch+1} " +
            f"({samples_trained}/{total_samples}): " +
            f"Loss={avg_loss:.5f}, Accuracy={accuracy:.5f}")
    

    # >>> INSERT: final static plots & restore best model
  plt.ioff()
  plt.figure(figsize=(10,4))
  # Loss
  plt.subplot(1,2,1)
  plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
  plt.plot(range(1, len(val_losses)  +1), val_losses,   label='Val Loss')
  plt.title('Final Loss vs Epoch')
  plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
  # Acc
  plt.subplot(1,2,2)
  plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
  plt.plot(range(1, len(val_accs)  +1), val_accs,   label='Val Acc')
  plt.title('Final Accuracy vs Epoch')
  plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
  # plt.show()
  ts = datetime.datetime.now().isoformat(timespec="seconds")
  fname = f"epoch-loss-acc_{ts}.png".replace(":", "-")
  plt.savefig(fname, dpi=150)
  print(f"Epoch-Loss-Acc saved as {fname}")

  # reload the best weights
  model.load_state_dict(torch.load('best_model.pt'))


  return batch_size, len(filepaths)

def test(model, filepaths, labels, device, image_dimensions,
         train_batch, train_inputs, epochs, save_cm=False, model_notes=""):
  batch_size = 60
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
      batch_inputs = load_images(filepaths[i : i + batch_size], device, dimensions=image_dimensions, augmentation="224")
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


class FruitCNN(nn.Module):
  def __init__(self, input_channels, image_dimensions):
    super(FruitCNN, self).__init__()
    self.image_dimensions = image_dimensions
    # in_channels = no. of input channels
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
    # in_channels=16 because our out_channels=16 from previous layer.
    # out_channels=32 means we are using 32 filters, each filter of size 3x3x16,
    # in this layer.
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    # Max Pooling Layer: downsample by a factor of 2.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # Fully Connected Layer 1: input size = 7 * 7 * 32 (from feature maps), output size = 128.
    self.fc1 = nn.Linear(in_features= self.conv2.out_channels * (int(self.image_dimensions[0] / 4)) * (int(self.image_dimensions[1] / 4)), out_features=128)
    # self.fc1 = nn.Linear(in_features= self.conv2.out_channels * (image_dimensions[0]/4) * (image_dimensions[1]/4), out_features=128)
    
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
    x = x.view(-1, self.conv2.out_channels * (int(self.image_dimensions[0] / 4)) * (int(self.image_dimensions[1] / 4)))

    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x
  

class FruitCNNWiderFilters(nn.Module):
  def __init__(self, input_channels, image_dimensions):
    super(FruitCNNWiderFilters, self).__init__()
    self.image_dimensions = image_dimensions
    # in_channels = no. of input channels
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
    # in_channels=16 because our out_channels=16 from previous layer.
    # out_channels=32 means we are using 32 filters, each filter of size 3x3x16,
    # in this layer.
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    # Max Pooling Layer: downsample by a factor of 2.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # Fully Connected Layer 1: input size = 7 * 7 * 32 (from feature maps), output size = 128.
    self.fc1 = nn.Linear(in_features= self.conv2.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4), out_features=128)
    # self.fc1 = nn.Linear(in_features= self.conv2.out_channels * (image_dimensions[0]/4) * (image_dimensions[1]/4), out_features=128)
    
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
    x = x.view(-1, self.conv2.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4))

    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x
  


class MnistCNN(nn.Module):
  def __init__(self, input_channels, image_dimensions):
    super(MnistCNN, self).__init__()
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
    self.fc1 = nn.Linear(self.conv2.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4), out_features=128)
    
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
    x = x.view(-1, self.conv2.out_channels * int(self.image_dimensions[0] / 4) * int(self.image_dimensions[1] / 4))

    # Fully connected layers
    x = self.fc1(x)
    x = self.relu(x)
    
    # Output layer (no activation since we apply softmax in the loss function)
    x = self.fc2(x)
    
    return x