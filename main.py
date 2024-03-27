import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from helper_functions import*
from dataclass import*
from modelClasses import*
from loss_fn import*
import segmentation_models_pytorch as smp


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def main():



    
    #set up parameters
    # num_classes=1 + 1 #including background
    #experiment for other label
    num_classes=1 + 1 + 1#including other and background

    # Initialize your segmentation dataset (replace with your dataset class)
    dataset = SemanticDataset(classes_threshold=num_classes,status="train") #not plus one since starts at 0

    # Set up data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True) #TODO: check num of workers

    # Define the segmentation model (replace with your preferred architecture)
    # modelseg = YoloSemSkipn(numClasses=num_classes) #+ 1 since we need 81 channels
    modelseg=End2end(numClasses=num_classes)
    # modelseg = torch.load("/home/jawad/codes/YoloPan/trained_models/YoloSemSkipn_bn_fenceother_allparm_epochs1.pth").eval()
    #experiment: try unet 
    # modelseg = smp.Unet(
    #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=num_classes,                      # model output channels (number of classes in your dataset)
    # )
    # for name,param in modelseg.named_parameters():
    #     print(name)
    #     param.requires_grad=True
    # modelseg = torch.load("/home/jawad/codes/YoloPan/trained_models/YoloSemSkipn_bn_stairshouse_allparm_unet_myresize_epochs9.pth").eval()


    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelseg = modelseg.to(device)
    print("segmodel :")
    torchinfo.summary(modelseg)
    

    
    # Define loss function and optimizer
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss(alpha=1, gamma=2)

    # criterion=DiceLoss()
    
    optimizer = optim.SGD(modelseg.parameters(), lr=0.01, momentum=0.9)

    # Set the number of training epochs
    num_epochs = 25

    # Training loop
    print(f'dataset size {len(dataset.img_names_list)}')
    train_semantic_segmentation(modelseg, data_loader, criterion, optimizer,num_epochs, device)
    # inference_semantic_segmentation(modelseg, data_loader,device)
    # torchinfo.summary(model,(1,64,80,80))


if __name__=="__main__":
    main()
