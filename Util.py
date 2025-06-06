import os, torch
import numpy as np
from torchvision import transforms
from PIL import Image

from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_saturation
from torchvision.transforms.functional import adjust_hue
from torchvision.transforms import functional as F


class PadToSquare:
    """Pad PIL image to a square, keeping content centred."""
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size                       # width, height
        max_wh   = max(w, h)                  # target square side
        pad_left = (max_wh - w) // 2
        pad_top  = (max_wh - h) // 2
        pad_right  = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        # (left, top, right, bottom)
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom),
                     fill=0, padding_mode="constant")


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
  #transform = transforms.Compose([
  #  transforms.Resize((28, 28)),
  #  transforms.ToTensor()
  #  ])
  transform = transforms.Compose([
    transforms.Resize(160),
    PadToSquare(),
    transforms.Resize((160, 160)),      # shorter edge → 160, keeps aspect
    # transforms.CenterCrop(dimensions),
    transforms.ToTensor()       # convert to [0,1] tensor
    ])
  
  # Store tensor list for all images 
  tensor_list = []
 
  # Perform data conversion into tensors
  for image_path in filepaths:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    tensor_list.append(image_tensor.unsqueeze(0).to(device))
    
  # Check if batch of tensors have been properly added to the list, else return empty tensor
  if tensor_list:
    result_tensor = torch.cat(tensor_list, dim=0)
    return result_tensor
  else:
    #return torch.empty(0, 3, 28, 28, device=device)
    return torch.empty(0, 3, 160, 160, device=device)

# load_images(["./train/apple_1.jpg"], device="cuda")