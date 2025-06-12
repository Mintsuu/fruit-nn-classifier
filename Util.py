import os, torch
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_saturation
from torchvision.transforms.functional import adjust_hue
from torchvision.transforms import functional as F
from torchvision import transforms 
from PIL import Image

# LABEL_VEC = {
#     "apple"  : torch.tensor([1, 0, 0], dtype=torch.float32),
#     "orange" : torch.tensor([0, 1, 0], dtype=torch.float32),
#     "banana" : torch.tensor([0, 0, 1], dtype=torch.float32),
#     "mixed"   : torch.tensor([1, 1, 1], dtype=torch.float32),
# }


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
                     fill=255, padding_mode="constant")

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

  # # map each folder to its 3-bit vector
  # for class_name in class_names:
  #   key = class_name.split("_")[0]   
  #   labels.append(LABEL_VEC[key])

  # labels_tensor = torch.stack(labels).to(device)   # [N, 3] float32
  # return LABEL_VEC, np.array(class_to_idx), labels_tensor

def load_images(filepaths, device, dimensions, augmentation):
  # Define image transformation function

  # Minor 
  # transform = transforms.Compose([transforms.Resize(180), transforms.RandomResizedCrop(160,(0.9,1.0)),
  #               transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15,0.15,0.15,0.02),
  #               transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],
  #                        std =[0.229,0.224,0.225])])


  # Extreme 
  # transform = transforms.Compose([transforms.Resize(180), transforms.RandomResizedCrop(160,(0.8,1.0)),
  #               transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
  #               transforms.ColorJitter(0.25,0.25,0.25,0.05),
  #               transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],
  #                        std =[0.229,0.224,0.225]), transforms.RandomErasing(p=0.2)])
  

#   transforms.Compose([
#     transforms.RandomResizedCrop(size),
#     transforms.RandomHorizontalFlip(),           # flips
#     transforms.RandomRotation(degrees=15),       # small rotations
#     transforms.ColorJitter(
#         brightness=0.2, contrast=0.2, 
#         saturation=0.2, hue=0.05
#     ),                                           # lighting changes
#     transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
#     transforms.ToTensor()
# ])

  if augmentation == "test":
    transform = transforms.Compose([
      transforms.Resize(180),
      transforms.CenterCrop((160,160)),     # shorter edge → 160, keeps aspect
      # transforms.RandomCrop(dimensions),
      transforms.ToTensor()       # convert to [0,1] tensor
      ])
    
  elif augmentation == "test3":
    transform = transforms.Compose([
      transforms.Resize(160),
      PadToSquare(),
      transforms.Resize((160,160)),
      transforms.RandomHorizontalFlip(),           # flips
      transforms.RandomRotation(degrees=10),       # small rotations
      transforms.ColorJitter(
           saturation=0.1
       ),                                           # lighting changes
      transforms.ToTensor()
      ])

  elif augmentation == "john": 
    transform = transforms.Compose([
      transforms.Resize((180, 180)),  # Resize first to make random crop meaningful
      transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),  # Random crop + resize
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(degrees=15),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random')
  ])
    
  elif augmentation == "mnist":
    transform = transforms.Compose([
      transforms.Resize(28),
      transforms.CenterCrop((28,28)),     # shorter edge → 160, keeps aspect
      transforms.Grayscale(),
      transforms.ToTensor()
    ])

  elif augmentation == "224":
    transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop((224,224)),
      transforms.Resize((224,224)),     # shorter edge → 160, keeps aspect
      transforms.ToTensor()
    ])

  elif augmentation == "160":
    transform = transforms.Compose([
      transforms.Resize(160),
      transforms.CenterCrop((160,160)),    # shorter edge → 160, keeps aspect
      transforms.ToTensor()
    ])      
  


  # transform = transforms.Compose([
  #   transforms.Resize(180),
  #   PadToSquare(),     # shorter edge → 160, keeps aspect
  #   transforms.RandomResizedCrop(160,(0.9,1.0)),
  # transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15,0.15,0.15,0.02),
  #                transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],
  #                         std =[0.229,0.224,0.225])])
  
  
  # Store tensor list for all images 
  tensor_list = [] # empty array where will add each batch tensor into array 
 
  # Perform data conversion into tensors
  for image_path in filepaths:
    if not os.path.isfile(image_path):
        continue
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
    
    elif augmentation == "train":
      sat  = adjust_saturation(image_tensor, saturation_factor=1.5)
      # hue  = adjust_hue(image_tensor,   hue_factor=0.1)
      stack = torch.stack([image_tensor, sat], dim=0)
      tensor_list.append(stack.to(device))

    elif augmentation == "test":
      tensor_list.append(image_tensor.unsqueeze(0).to(device))  # [1, 3, H, W]

    elif augmentation == "mnist":
      tensor_list.append(image_tensor.unsqueeze(0).to(device))  # [1, 3, H, W]

    elif augmentation == "224":
      tensor_list.append(image_tensor.unsqueeze(0).to(device))  # [1, 3, H, W]

    else:
      tensor_list.append(image_tensor.unsqueeze(0).to(device))

    
  # Check if batch of tensors have been properly added to the list, else return empty tensor
  if tensor_list:
    result_tensor = torch.cat(tensor_list, dim=0)
    return result_tensor
  else:
    print("No images loaded into tensors ")
    return torch.empty(0, 3, 28, 28, device=device)

# load_images(["./train/apple_1.jpg"], device="cuda")


