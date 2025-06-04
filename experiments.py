from torchvision import torch
from SimpleNN import SimpleCNN, test
from Util import prepare_data


target_image_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = SimpleCNN(input_channels=7, image_dimensions=target_image_size).to(device)

path_to_saved_state = "./model/model_saved_state"
loaded_model.load_state_dict(torch.load(path_to_saved_state, map_location=device))
loaded_model.eval()

print(f"Model loaded successfully.")

print(f"Running tests with loaded model...")
dir_test = "./test/"
_, filepaths, labels = prepare_data(dir_test, device)
test(loaded_model, filepaths, labels, device, image_dimensions=target_image_size)

print(f"Testing completed!")
