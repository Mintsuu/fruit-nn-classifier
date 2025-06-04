import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Input and output folder paths
input_dir = 'input_images'       # The folder where the original image is located
output_dir = 'augmented_images'  # Folder for saving the transformed image


# Define transformations
def get_transform():
    return transforms.Compose([
        transforms.RandomRotation(45),                                                # Randomly rotate ±45 degrees
        transforms.RandomHorizontalFlip(p=0.5),                                  # 50% probability level flip
        transforms.RandomVerticalFlip(p=0.5),                                       # 50% probability of vertical flipping
        transforms.RandomResizedCrop(224, scale=(0.5, 1.5)),               #Random scaling and cropping to 224x224
    ])


# Generate several enhanced images for each picture
num_copies = 2


# Traverse all the images in the folder save them
for fname in tqdm(os.listdir(input_dir)):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img = Image.open(os.path.join(input_dir, fname)).convert('RGB')
        base = os.path.splitext(fname)[0]

        for i in range(num_copies):
            transformed = transform(img)
            save_path = os.path.join(output_dir, f'{base}_aug{i}.jpg')
            transformed.save(save_path)

            score = sharpness_score(transformed)
            print(f"Saved: {save_path}, Sharpness Score: {score:.2f}")





