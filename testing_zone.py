'''
test the yolov8-n
'''
import torch 
import torch.nn as nn
from mmdet.apis import init_detector, inference_detector
import torchinfo 

confg_file="/home/jawad/codes/YoloPan/yolo_configs/yolov8_n_mask-refine_syncb.py"
checkpoint_file="/home/jawad/codes/YoloPan/yolo_models/yolov8_n_mask-refine.pth"

model = init_detector(confg_file, checkpoint_file, device='cuda')  # or device='cuda:0'
# Register a forward hook on the preprocess section
print(model)
def forward_hook(module, input, output):
    print(f"Forward hook called for {module.__class__.__name__}")
    print(f"Input : {input}")
    print(f"Output : {output.shape}")
    print("===")
hook_handle = model.neck.top_down_layers[1].register_forward_hook(forward_hook)

#inference
#edit te model by changing the parameters 
#but how to pass the labels during inference
# out=inference_detector(model, '/home/jawad/codes/YoloPan/demo_images/demo.jpg')
torchinfo.summary(model,(1,3,640,640))
# print(model)
model(torch.rand(1,3,640,640).to('cuda'))
#how much deos it cost to upsample to original image shape 
a=torch.rand(1,64,80,80)

# print(model.data_preprocessor(torch.rand(1,3,600,600)))



class modeltrans(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(64, 80, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(80, 80, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(80, 80, 4, stride=2, padding=1)

        self.relu= nn.ReLU()


    def forward(self,x):
        return self.upsample3(self.upsample2(self.upsample1(x)))


m=modeltrans()
# torchinfo.summary(m,(1,64,80,80))

# print(model)

model.neck.reducelayers=nn.ReLU()

# print(model)
# print(model.neck.reducelayers)