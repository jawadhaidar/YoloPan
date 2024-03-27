import torch 
import torch.nn as nn
import torchinfo 
import numpy as np
import torch.nn.functional as F
from mmdet.apis import init_detector, inference_detector




class YoloSem(nn.Module):
    '''
    Adds a semantic block to yolov8
    '''
    def __init__(self,numClasses):
        super().__init__() #attribtes from nn.Module

        self.numClasses=numClasses
        self.upsampleP32= nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1) #64 for nano #320 x
        self.refineP32=nn.Conv2d(64,64,3,stride=1,padding=1)
        self.upsampleP21= nn.ConvTranspose2d(64, 80, 4, stride=2, padding=1)
        self.refineP21=nn.Conv2d(80,80,3,stride=1,padding=1)
        self.upsampleP1I= nn.ConvTranspose2d(80, 80, 4, stride=2, padding=1) 
        self.refineP1I=nn.Conv2d(80,80,3,stride=1,padding=1)
        self.segment=nn.Conv2d(80, self.numClasses, 3, stride=1, padding=1) 
        self.relu=nn.ReLU()
        # self.softmax=nn.Softmax(dim=1)
        #init model



    def forward(self,p3):
        #yolo inference with postprocess
        #x here is the path to the image
        #upscale
        out=self.refineP32(self.relu(self.upsampleP32(p3)))
        out=self.refineP21(self.relu(self.upsampleP21(out)))
        out=self.relu(self.refineP1I(self.relu(self.upsampleP1I(out))))#should reach here image size
        #segment 
        out=self.segment(out) #cross entropy loss does the softmax

        return out 
    
class YoloSemSkip(nn.Module):
    '''
    Adds a semantic block to yolov8
    '''
    def __init__(self,numClasses):
        super().__init__() #attribtes from nn.Module

        self.numClasses=numClasses
        # self.PPM=PyramidPoolingModule(in_channels=320, out_channels=160, pool_sizes = [1, 2, 3, 6])
        self.upsampleP32= nn.ConvTranspose2d(320, 160, 4, stride=2, padding=1) #64 for nano #320 x
        self.refineP32=nn.Conv2d(160,160,3,stride=1,padding=1)
        self.upsampleP21= nn.ConvTranspose2d(160, 80, 4, stride=2, padding=1)
        self.refineP21=nn.Conv2d(80,80,3,stride=1,padding=1)
        self.upsampleP1I= nn.ConvTranspose2d(80, 80, 4, stride=2, padding=1) 
        self.refineP1I=nn.Conv2d(80,80,3,stride=1,padding=1)

        self.segment=nn.Conv2d(80, self.numClasses, 3, stride=1, padding=1) 
        self.relu=nn.ReLU()
        # self.softmax=nn.Softmax(dim=1)
        #init model



    def forward(self,p3,p2,p1):
        #yolo inference with postprocess
        #x here is the path to the image
        #upscale
        # p3=self.PPM(p3)
        out=self.refineP32(self.relu(self.upsampleP32(p3))+p2)
        out=self.refineP21(self.relu(self.upsampleP21(out))+p1)
        out=self.relu(self.refineP1I(self.relu(self.upsampleP1I(out))))#should reach here image size
        #pool
        # out=self.PPM(out)
        #segment 
        out=self.segment(out) #cross entropy loss does the softmax

        return out 
    
class YoloSemSkipn(nn.Module):
    '''
    Adds a semantic block to yolov8
    '''
    def __init__(self,numClasses):
        super().__init__() #attribtes from nn.Module

        self.numClasses=numClasses
        # self.PPM=PyramidPoolingModule(in_channels=320, out_channels=160, pool_sizes = [1, 2, 3, 6])
        self.upsampleP32= nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) #64 for nano #320 x
        self.refineP32=nn.Conv2d(32,32,3,stride=1,padding=1)
        self.BN32=nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.upsampleP21= nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.refineP21=nn.Conv2d(16,16,3,stride=1,padding=1)
        self.BN21=nn.BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.upsampleP1I= nn.ConvTranspose2d(16, 80, 4, stride=2, padding=1) 
        self.refineP1I=nn.Conv2d(80,80,3,stride=1,padding=1)
        self.BN1I=nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.segment=nn.Conv2d(80, self.numClasses, 3, stride=1, padding=1) 
        self.relu=nn.ReLU()
        # self.softmax=nn.Softmax(dim=1)
        #init model

        #TODO: why you did not add batch norm

    def forward(self,p3,p2,p1):
        #yolo inference with postprocess
        #x here is the path to the image
        #upscale
        # p3=self.PPM(p3)
        #TODO: print max min of self.upsampleP32(p3)) and p2
        # print("p3 and p2 max min min")
        # print(self.upsampleP32(p3).max(),p2.max(),self.upsampleP32(p3).min(),p2.min())
        out=self.BN32(self.refineP32(self.relu(self.upsampleP32(p3))+p2))
        out=self.BN21(self.refineP21(self.relu(self.upsampleP21(out))+p1))
        out= self.relu(self.BN1I(self.refineP1I(self.refineP1I(self.relu(self.upsampleP1I(out))))))#should reach here image size
        #pool
        # out=self.PPM(out)
        #segment 
        out=self.segment(out) #cross entropy loss does the softmax

        return out 
    
class YoloSemPPM(nn.Module):
    '''
    Adds a semantic block to yolov8
    '''
    def __init__(self,numClasses):
        super().__init__() #attribtes from nn.Module

        self.numClasses=numClasses
        self.PPM=PyramidPoolingModule(in_channels=320, out_channels=80, pool_sizes = [1, 2, 3, 6])
        #upsammple    

   
        self.segment=nn.Conv2d(80, self.numClasses, 3, stride=1, padding=1) 
        self.relu=nn.ReLU()
        # self.softmax=nn.Softmax(dim=1)
        #init model



    def forward(self,f):
        #yolo inference with postprocess
        #x here is the path to the image
        #upscale
        out=self.PPM(f)
        #upsample
        upsampled_output = F.interpolate(out, size=(640,640), mode='bilinear', align_corners=False)
        #segment 
        out=self.segment(upsampled_output) #cross entropy loss does the softmax

        return out 
    


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size) for output_size in pool_sizes
        ])
        # Adjust the number of output channels for the convolutional layer
        self.conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), out_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for pool_layer in self.pool_layers:
            pooled_features = pool_layer(x)
            features.append(F.interpolate(pooled_features, size=x.size()[2:], mode='bilinear', align_corners=False))
        output = torch.cat(features, dim=1)
        output = self.conv(output)
        return output

    
class End2end(nn.Module):
    def __init__(self,numClasses):
        super().__init__()
        yoloConfigPath="/home/jawad/codes/YoloPan/yolo_configs/yolov8_n_mask-refine_syncb.py"
        yoloModelPath="/home/jawad/codes/YoloPan/yolo_models/yolov8_n_mask-refine.pth"
        # yoloConfigPath="/home/aub/codes/YoloPan/yolo_configs/YOLOv8-x.py"
        # yoloModelPath="/home/aub/codes/YoloPan/yolo_models/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth"
        self.yoloModel=init_detector(yoloConfigPath, yoloModelPath, device='cuda')
        # self.yoloModel(torch.rand(1,3,640,640).to('cuda'))
        #change the right parameters to training model
        for name,module in self.yoloModel.named_children():
            print(name)
            if name =='bbox_head':
                for _, param in module.named_parameters():
                    param.requires_grad = False
            else:
                for _, param in module.named_parameters():
                    param.requires_grad = True


        #break it down
        self.dataprocess=self.yoloModel.data_preprocessor
        #test
        self.backbone=self.yoloModel.backbone
        self.neck=self.yoloModel.neck 
        self.bbox_head=self.yoloModel.bbox_head
        #segmentation model
        self.seg=YoloSemSkipn(numClasses).to('cuda')

    def forward(self,x):
        #dataprocess
        x=self.dataprocess(x)
        input=x['inputs']
        #backbone sequential
        stem=self.backbone.stem(input)
        stage1=self.backbone.stage1(stem)
        stage2=self.backbone.stage2(stage1)
        stage3=self.backbone.stage3(stage2)
        stage4=self.backbone.stage4(stage3)
       
        #neck
        out=self.neck((stage2,stage3,stage4))
        #boxhead
        # outbox=self.bbox_head(out)
        #semhead
        out=self.seg(out[0],stage1,stem) #0 or 2
        del stem
        del stage1
        del stage2
        del stage3
        del stage4
        return out



        
if __name__=="__main__":

    model=End2end(numClasses=3)

    torchinfo.summary(model)
    result=model({'inputs':torch.rand(1,3,640,640).to('cuda')})
    # print(result)
    # print(model.backbone)
    # print(model.backbone.stage4)
