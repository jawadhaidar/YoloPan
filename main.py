import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from helper_functions import*
from dataclass import*
from modelClasses import*
from loss_fn import*

activation={}
def forward_hook(module, input, output):
    # print(f"Forward hook called for {module.__class__.__name__}")
    # print(f"Input : {input}")
    # print(f"Output shape : {output.shape}")
    # print("===")
    activation['out']=output

activation_img={}
def forward_hook_img(module, input, output):
    # print(f"Forward hook called for {module.__class__.__name__}")
    # print(f"Input : {input}")
    # print(f"Output shape : {output}")
    # print("===")
    activation_img['inimg']=input#input[0]['inputs'][0]
    activation_img['outimg']=output

def forward_hook_stem(module, input, output):
    # print(f"Forward hook called for {module.__class__.__name__}")
    # print(f"Input : {input}")
    # print(f"Output shape : {output.shape}")
    # print("===")
    activation['outstem']=output

def forward_hook_stage1(module, input, output):
    # print(f"Forward hook called for {module.__class__.__name__}")
    # print(f"Input : {input}")
    # print(f"Output shape : {output.shape}")
    # print("===")
    activation['outstage1']=output

def forward_hook_stage4(module, input, output):
    # print(f"Forward hook called for {module.__class__.__name__}")
    # print(f"Input : {input}")
    # print(f"Output shape : {output.shape}")
    # print("===")
    activation['outstage4']=output


def main():

    #set up yolo 
    yoloConfigPath="/home/jawad/codes/YoloPan/yolo_configs/yolov8_n_mask-refine_syncb.py"
    yoloModelPath="/home/jawad/codes/YoloPan/yolo_models/yolov8_n_mask-refine.pth"
    # yoloConfigPath="/home/aub/codes/YoloPan/yolo_configs/YOLOv8-x.py"
    # yoloModelPath="/home/aub/codes/YoloPan/yolo_models/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth"
    yoloModel=init_detector(yoloConfigPath, yoloModelPath, device='cuda')
    # Freeze all parameters in the model
    for param in yoloModel.parameters():
        param.requires_grad = False
    #register hook
    # print(yoloModel)
    yoloModel.data_preprocessor.register_forward_hook(forward_hook_img)
    yoloModel.neck.top_down_layers[1].register_forward_hook(forward_hook)
    yoloModel.backbone.stem.register_forward_hook(forward_hook_stem)
    yoloModel.backbone.stage1.register_forward_hook(forward_hook_stage1)
    yoloModel.backbone.stage4.register_forward_hook(forward_hook_stage4)

    
    #set up parameters
    num_classes=91 + 1 #including background

    # Initialize your segmentation dataset (replace with your dataset class)
    dataset = SemanticDataset(classes_threshold=num_classes,status="train") #not plus one since starts at 0

    # Set up data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True) #TODO: check num of workers

    # Define the segmentation model (replace with your preferred architecture)
    # modelseg = YoloSemSkipn(numClasses=num_classes) #+ 1 since we need 81 channels
                    
    modelseg = torch.load("/home/jawad/codes/YoloPan/trained_models/YoloSemSkipn_bn_stuffall_epochs7.pth")

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelseg = modelseg.to(device)
    print("segmodel :")
    torchinfo.summary(modelseg)
    

    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # criterion=DiceLoss()
    
    optimizer = optim.SGD(modelseg.parameters(), lr=0.01, momentum=0.9)

    # Set the number of training epochs
    num_epochs = 100

    # Training loop
    print(f'dataset size {len(dataset.img_names_list)}')
    train_semantic_segmentation(yoloModel, modelseg, data_loader, criterion, optimizer,num_epochs,activation,activation_img, device)
    # inference_semantic_segmentation(modelseg, data_loader,device)
    # torchinfo.summary(model,(1,64,80,80))


if __name__=="__main__":
    main()
