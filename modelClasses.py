import torch 
import torch.nn as nn
import torchinfo 
import numpy as np
import torch.nn.functional as F




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

    

        
if __name__=="__main__":

    # sem=YoloSem(80).to(device="cuda")
    # torchinfo.summary(sem,(1,320,80,80))
   # Example usage of the Pyramid Pooling Module
    in_channels = 256  # Number of input channels
    out_channels = 128  # Number of output channels
    pool_sizes = [1, 2, 3, 6]  # List of pooling kernel sizes (or output sizes for adaptive pooling)

    # Initialize the Pyramid Pooling Module with the example parameters
    ppm = PyramidPoolingModule(in_channels, out_channels, pool_sizes)

    # Generate example input feature map
    batch_size = 4
    height = 64
    width = 64
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass through the Pyramid Pooling Module
    output = ppm(x)

    # Print the shape of the output feature map
    print("Output shape:", output.shape)
    #upsampel 
    upsampled_output = F.interpolate(output, size=(640,640), mode='bilinear', align_corners=False)
    print("Outupsampled_outputput shape:", upsampled_output.shape)
    model=YoloSemPPM(numClasses=3)
    o=model(x)
    print(o.shape)
    torchinfo.summary(model,(1,256,64,64))