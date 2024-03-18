import torch 
from dataclass import *
import torchinfo 

#load dataset 
#set up parameters
num_classes=80

# Initialize your segmentation dataset (replace with your dataset class)
dataset = SemanticDataset(classes_threshold=num_classes) #TODO: add a validation status

# Set up data loader
data_loader = DataLoader(dataset, batch_size=1, shuffle=True) #TODO: check num of workers

# Define the segmentation model (replace with your preferred architecture)
model = torch.load("/home/jawad/codes/YoloPan/trained_models/model_epochs2.pth")

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(model)
torchinfo.summary(model,(1,64,640,640))



