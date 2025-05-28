import os, torch
import numpy as np
from torchvision import transforms
from PIL import Image


def dataset_labels(labels):
  label_list = list(set(labels))
  print(label_list)

def print_named_params(model):
  for name, param in model.named_parameters():
    print(f"{name}: {param.numel()}")

# This is not used, since we're not storing our classified images
# in separate folders

def load_filepaths(target_dir): 
  paths = []
  print(target_dir)
  files = os.listdir(target_dir)
  for file in files:
    paths.append(f"{target_dir}/{file}")
  return paths


def prepare_data(target_dir):
  labels = []
  # Retrieves the names of all files in the target directory
  class_names = os.listdir(target_dir)
  # Removing gitignore file from dataset
  class_names.remove(".gitignore")
  # TODO: Perform data cleansing here (e.g. renaming of misclassified dataset)
  # Creating a dict, using the index as the key and the filename as the value
  # class_to_idx = [name: idx for idx, name in enumerate(class_names)]
  class_to_idx = [f"{target_dir}/{name}" for name in class_names]
  # Substring out the name of the fruit from the filename
  for class_name in class_names:
    labels.append(class_name.split("_")[0])
  # Binning each possible outcome into a dictionary
  # 0 = banana
  # 1 = orange
  # 2 = apple
  # 3 = mixed
  binned_labels = { name: index for index, name in enumerate(sorted(set(labels))) }
  # Converting categories into their assigned numerical bins
  binned_outputs = [binned_labels[name] for name in labels]
  return binned_labels, np.array(class_to_idx), torch.tensor(binned_outputs)

def load_images(filepaths):
  transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
    ])
  
  # print(transform)
  

  # Instantiate class to transform image to tensor
  to_tensor = transforms.ToTensor()

  tensor = None

  # List all files in the directory
  for item in filepaths:
    #print("Opening:", item)
    image = Image.open(item).convert("RGB")
    #print(f"image size = {image.size}")

    # transforms.ToTensor() performs transformations on images
    # values of img_tensor are in the range of [0.0, 1.0]
    img_tensor = transform(image)

    #img_tensor = to_tensor(image) # convert into pytorch's tensor to work with
    #print(f"img_tensor.shape = {img_tensor.shape}")
    #input()

    if tensor is None:
      # size: [1,1,28,28]
      tensor = img_tensor.unsqueeze(0) # add a new dimension
    else:
      # concatenate becomes [2,1,28,28], [3,1,28,28], [4,1,28,28] ...
      # dim=0 concatenates along the axis=0 (row-wise)
      tensor = torch.cat((tensor, img_tensor.unsqueeze(0)), dim=0)
    
  return tensor

load_images(["./train/apple_1.jpg"])