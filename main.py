import torch
import torch.nn as nn
import torch.optim as optim
from SimpleNN import SimpleCNN, train, test
from Util import prepare_data

### EXPERIMENT PARAMS ###

N_EPOCHS = 6
BATCH_SIZE = 16
IMAGE_DIMS = (160, 160)

#########################

# Set device type (check if you have a NVIDIA card)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using {device}...")
dir_train = "./train" 
output_labels, filepaths, labels = prepare_data(dir_train, device)

# Instantiate the model, define the loss function and optimizer
model = SimpleCNN(input_channels=3, image_dimensions=IMAGE_DIMS).to(device)

# Train the model
criterion = nn.CrossEntropyLoss() # define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_batch, train_total = train(model, criterion, optimizer, filepaths, labels, device, epochs=N_EPOCHS, image_dimensions=IMAGE_DIMS, augment = "test")

# Test the model
dir_test = "./test/"
_, filepaths, labels = prepare_data(dir_test, device)
test(model, filepaths, labels, device, image_dimensions=IMAGE_DIMS, train_batch=train_batch, train_inputs=train_total, epochs = N_EPOCHS, save_cm=True)