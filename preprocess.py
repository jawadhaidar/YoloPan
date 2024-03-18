from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.registry import MODELS, TASK_UTILS
import torch
#method1: build from dict
data_preprocess=dict(
    # type='YOLOv5DetDataPreprocessor',
    type='DetDataPreprocessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True)

# preprocess_built=TASK_UTILS.build(data_preprocessor)

# preprocess_built(torch.rand(1,3,400,640))

#method2
preprocess=DetDataPreprocessor(mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True) #this takes a dictionary 

