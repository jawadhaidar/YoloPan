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



torchinfo.summary(model,(1,3,640,640))
# print(model)
result=model(torch.rand(1,3,640,640).to('cuda'))
# print(result)
print(model)
# for name, module in model.backbone.named_children():
#     print(name,module)