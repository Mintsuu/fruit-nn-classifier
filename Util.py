import os, torch
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast, adjust_saturation, adjust_sharpness
from torchvision.transforms.functional import adjust_hue
from torchvision.transforms import functional as F
from torchvision import transforms 
from PIL import Image

fixed_training_labels = {
  'banana_35.jpg': 'mixed',
  'banana_61.jpg': 'mixed',
  'orange_62.jpg': 'ignore',
  }
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

  if target_dir == './train':
    for blacklist in fixed_training_labels:
      if fixed_training_labels[blacklist] == 'ignore':
        print(blacklist)
        class_names.remove(blacklist)
  # Creating a dict, using the index as the key and the filename as the value
  # class_to_idx = [name: idx for idx, name in enumerate(class_names)]
  class_to_idx = [f"{target_dir}/{name}" for name in class_names]
  # Substring out the name of the fruit from the filename
  print(class_names)
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

def load_images(filepaths, device, dimensions, augmentation):
  # Define image transformation function
  transform = transforms.Compose([
    transforms.Resize(160),
    # PadToSquare(),     # shorter edge → 160, keeps aspect
    transforms.CenterCrop(dimensions),
    transforms.ToTensor()       # convert to [0,1] tensor
    ])
  
  # Store tensor list for all images 
  tensor_list = [] # empty array where will add each batch tensor into array 
 
  # Perform data conversion into tensors
  for image_path in filepaths:
    combined_tensor = set_input_channels(image_path, dimensions)
    tensor_list.append(combined_tensor.unsqueeze(0).to(device))
   
    image = Image.open(image_path).convert("RGB")
    # pil image converted to tensor rgb 
    image_tensor = transform(image)


    save_image(image_tensor, "baseImage.png")


    # #  # apply saturation 
    # image_saturation_tensor = adjust_saturation(img=image_tensor, saturation_factor=2.0)
    # # #apply hue 
    # image_hue_tensor = adjust_hue(img = image_tensor, hue_factor = 0.2)
    # save_image(image_tensor, "baseline_test.png")
    # save_image(image_saturation_tensor, "saturation_test.png")
    # save_image(image_hue_tensor, "hue_test.png")
    
    # # add up the image_tensor with image_saturation_tensor
    # # combined_tensor = torch.cat(( image_saturation_tensor, image_hue_tensor), dim=0) 
    # # tensor_list.append(combined_tensor.unsqueeze(0).to(device))

    # # stack instead of cat, and along a NEW batch axis
    # aug_batch = torch.stack([image_tensor,
    #                      image_saturation_tensor,
    #                      image_hue_tensor], dim=0)    # shape [3, 3, D, D]

    # tensor_list.append(aug_batch.to(device))               # NO unsqueeze needed

    if augmentation == "sathue":
# create two extra colour-jittered copies
      sat  = adjust_saturation(image_tensor, saturation_factor=1.5)
      hue  = adjust_hue(image_tensor,   hue_factor=0.1)
      stack = torch.stack([image_tensor, sat, hue], dim=0)   # [3, 3, H, W]
      tensor_list.append(stack.to(device))
    

    elif augmentation == "test":
      tensor_list.append(image_tensor.unsqueeze(0).to(device))  # [1, 3, H, W]

    
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

