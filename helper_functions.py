import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import time
from mmdet.apis import init_detector, inference_detector
from torchvision import transforms
import matplotlib.pyplot as plt
import os



def train_semantic_segmentation(model,modelseg, train_loader, criterion, optimizer,num_epochs,activation,activation_img, device):
    # model.train() changes the output of inference
    epochs_loss_list=[]
    for epoch in range(num_epochs):
        epoch_loss=0
        for idbatch,batch_sample in enumerate(train_loader):
            imgs_batch_paths, masks =  batch_sample
            masks = masks.to(torch.long) #.to(torch.float32)
            masks=masks.to(device="cuda")
            # print(f'masks unique {torch.unique(masks)}')
            #perform detction 
            inference_detector(model, imgs_batch_paths)
            #get p3 from activation
            p1=activation['outstem']
            p2=activation['outstage1']
            p3=activation['out']
            f=activation['outstage4']
            # print(f'p3 shape {p3.shape}')
            img=activation_img['inimg']
            # print("shapes p1 2 3")
            # print(p1.shape,p2.shape,p3.shape)
            # print(f'img {img} shape {img.shape}')
            draw_img(img)
            # img=activation_img['out']['inputs']
            # print(f'img out shape {img.shape}')
            # draw_img(img.squeeze(0))

            # Forward pass on p3
            outputs = modelseg(p3,p2,p1) #640x640xnum_classes
            # Calculate loss #.to(torch.long)
            # print(outputs.shape)
            for mask in masks:
                    visualize_predicted_masks(mask)
            postprocess_seg(outputs,None)
            # print(f'outputs {outputs.shape} masks {masks.shape}')

        #     loss = criterion(outputs, masks) #make sure background channel is correct

        #     epoch_loss+=loss
        #     # Backward pass and optimization
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print(f'epoch {epoch} idbatch: {idbatch} batch loss {loss} epochs loss {epochs_loss_list}')
        # epochs_loss_list.append(epoch_loss)
        # print(f'epoch {epoch} results epochs loss {epochs_loss_list}')
        # time.sleep(10) 
        # # Save the trained model (optional)
        # if epoch>0:
        #      #delete previous .pth
        #      os.remove(f'/home/aub/codes/YoloPan/trained_models/YoloSemSkipn_bn_cloudstree_epochs{epoch-1}.pth')

        # torch.save(modelseg, f'/home/aub/codes/YoloPan/trained_models/YoloSemSkipn_bn_cloudstree_epochs{epoch}.pth')
        

def inference_semantic_segmentation(model, val_loader,device):

    for idbatch,batch_sample in enumerate(val_loader):
        imgs_batch_paths, masks =  batch_sample
        print(imgs_batch_paths)
        print(masks.shape)
        masks = masks.to(torch.long) #.to(torch.float32)
        masks=masks.to(device=device)
        visualize_predicted_masks(masks[0])
        #perform detction 
        model.detection_forward_with_post(imgs_batch_paths)
        #get p3 from activation
        p3=model.activation['out']
        # Forward pass on p3
        model = torch.load("/home/aub/codes/YoloPan/trained_models/model.pth")
        model.activation['out']=p3
        outputs = model(p3) #640x640xnum_classes
        postprocess_seg(outputs)

def postprocess_seg(masks_logits,img):
    print(masks_logits.shape)
    softmax=nn.Softmax(dim=0)
    #for each mask
    print(f'mask_logits shape {masks_logits.shape}')
    for mask_logits in masks_logits:
        print(mask_logits.shape)
        #apply softmax
        #apply argmax 
        mask=torch.argmax(softmax(mask_logits), dim=0)
        print(mask)
        visualize_predicted_masks(mask,img)
        

def visualize_predicted_masks(mask,img=None):
        if img!=None:
            tensor_to_pil = transforms.ToPILImage()
            image_pil = tensor_to_pil(img)
            plt.imshow(image_pil, cmap='viridis')

        plt.imshow(mask.cpu(), cmap='viridis', alpha=0.9)
        plt.title(f'Segmentation Mask')
        plt.show()
# def visualize_png_masks(image_path, masks_folder):
#     # Load the original image
#     img = plt.imread(image_path)

#     # Get a list of PNG mask files in the masks folder
#     mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.png')]

#     # Overlay each mask on the image
#     for mask_file in mask_files:
#         # Load the PNG mask
#         mask_path = os.path.join(masks_folder, mask_file)
#         mask = np.array(Image.open(mask_path))

#         # Overlay the mask on the image
#         plt.imshow(img)
#         plt.imshow(mask, cmap='viridis', alpha=0.5)
#         plt.title(f'Segmentation Mask: {mask_file}')
#         plt.show()

def draw_img(image_tensor):
    #Convert tensor to a PIL image
    tensor_to_pil = transforms.ToPILImage()
    image_pil = tensor_to_pil(image_tensor)

    # Display the image using Matplotlib
    plt.imshow(image_pil)
    plt.axis('off')  # Turn off axis labels
    plt.show()