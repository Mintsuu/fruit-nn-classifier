import os, torch
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast, adjust_saturation, adjust_sharpness
from torchvision.transforms.functional import adjust_hue
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


def prepare_data(target_dir, device):
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
  return binned_labels, np.array(class_to_idx), torch.tensor(binned_outputs).to(device)

def load_images(filepaths, device, dimensions):
  # Define image transformation function
  
  # Store tensor list for all images 
  tensor_list = [] # empty array where will add each batch tensor into array 
 
  # Perform data conversion into tensors
  for image_path in filepaths:
    combined_tensor = set_input_channels(image_path, dimensions)
    tensor_list.append(combined_tensor.unsqueeze(0).to(device))
    
  # Check if batch of tensors have been properly added to the list, else return empty tensor
  if tensor_list:
    result_tensor = torch.cat(tensor_list, dim=0)
    return result_tensor
  else:
    print("No images loaded into tensors ")
    return torch.empty(0, 3, 28, 28, device=device)


def set_input_channels(image_url, dimensions):
  transform = transforms.Compose([
    transforms.Resize(dimensions),
    transforms.ToTensor()
  ])
  image = Image.open(image_url).convert("RGB")
  image_greyscale = Image.open(image_url).convert("L")
  image_tensor = transform(image)
  image_greyscale_tensor = transform(image_greyscale)
  # Experiment 1: Applying saturation to image tensor with a factor of 2.0
  image_saturation_tensor = adjust_saturation(img=image_tensor, saturation_factor=2.0)
  # Experiment 2: Applying hue adjustment to image tensor with a factor of 0.2
  image_hue_tensor = adjust_hue(img = image_tensor, hue_factor = 0.2)
  # Experiment 3: Applying greyscale adjustment to image tensor 
  image_greyscale_tensor = adjust_contrast(img = image_greyscale_tensor, contrast_factor=2)
  image_greyscale_tensor = adjust_sharpness(img = image_greyscale_tensor, sharpness_factor=2) 
  # Concat result tensors
  combined_tensor = torch.cat(( image_saturation_tensor, image_hue_tensor, image_greyscale_tensor), dim=0) 

  return combined_tensor

def save_test_images(image_name, image_tensors):
  num_channels = image_tensors.size(dim=0)
  for i in range(0, num_channels):
    save_image(image_tensors[i], f"./sample-images/{image_name}_channel_{i}.png" )

