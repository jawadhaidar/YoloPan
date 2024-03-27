import torchvision

from torchvision.datasets import CocoDetection
import os 
from torch.utils.data import Dataset, DataLoader
import re
import cv2 as cv
import numpy as np
import torch 
import torch.nn as nn
from mmdet.apis import init_detector, inference_detector
from torchvision.transforms import transforms
import mmcv
import torch.nn.functional as F
from mmyolo.datasets.transforms.transforms import LetterResize



def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class SemanticDataset(Dataset):
    #constructer
    def __init__(self,transform=None,status="train",classes_threshold=81):
        
        # self.img_path=f"/home/aub/mmdetection/data/coco/{status}2017" # config['DATA']['imgpath'] #
        # self.mask_path=f"/home/aub/datasets/stuffthingmaps_trainval2017/{status}2017" #config['DATA']['maskpath']
        self.img_path=f"/home/jawad/datasets/{status}sample2017/images" # config['DATA']['imgpath'] #
        self.mask_path=f"/home/jawad/datasets/{status}sample2017/semantic_labels" #config['DATA']['maskpath']
        self.img_names_list=sorted_alphanumeric(os.listdir( self.img_path))
        self.mask_names_list=sorted_alphanumeric(os.listdir( self.mask_path))
        self.num_imgs=len(self.img_names_list)
        #set transforms
        self.transform=transform
        self.classes_thr=classes_threshold #including background
        self.target_size=(640,640)
        self.LetterRseg=LetterResize((640,640),pad_val={'img':255},use_mini_pad=False,stretch_only=False,allow_scale_up=False,half_pad_param=False)
        self.LetterRimg=LetterResize((640,640),pad_val={'img':114},use_mini_pad=False,stretch_only=False,allow_scale_up=False,half_pad_param=False)
        #TODO: the difference in pad values might be confusing for the model
        '''
        pad val here might be missleading 
        it is used for mask not image 
        plus we are changing it in dataclass but thr to num_class-1
        '''

    def __len__(self):
        return len(self.img_names_list)
    
    def __getitem__(self, index):
        path_img=os.path.join(self.img_path,self.img_names_list[index])
        img=cv.cvtColor(cv.imread(path_img), cv.COLOR_BGR2RGB) 
        # print(img.shape)
        # mask=cv.imread(os.path.join(self.mask_path,self.mask_names_list[index]),cv.IMREAD_GRAYSCALE)
        mask=cv.imread(os.path.join(self.mask_path,self.mask_names_list[index]))
        # print(f'mask.shape {mask.shape}')

        # L._resize_img({"img":mask})
        #you might need to choose the first 80 classes 
        #TODO: fix the background label 
        # seg=self.LetterRseg._resize_img({"img":mask}) # this was possible after dding return results to the source code
        mask=self.LetterRseg._resize_img({"img":mask})["img"][:,:,0]
        #img
        # im=self.LetterRimg._resize_img({"img":img}) # this was possible after dding return results to the source code
        img=self.LetterRimg._resize_img({"img":img})['img']
        # print(img.shape,mask.shape)
        #experiment resize
        # img,mask=self.resize_image_and_mask(img, mask, (640,640))

      



        # print(f'unique before {np.unique(mask)}')
        mask[mask>=self.classes_thr-1]= self.classes_thr-1 #background was 255  0 1 2 3 4 5 6...79 (3)
        # mask=self.sem_resize(mask,(640,640)) 
        # print(f'unique  after {np.unique(mask)}')
        # mask=self.letterbox_mask(mask, self.target_size, fill_value=self.classes_thr-1)

        if self.transform is not None:
            #we are feeding this to their model 
            #theey do preprocessing internally 
            #we should transform only the mask 
            pass


        return img,mask
    
    def resize_image_and_mask(self,image, mask, new_size):
        """
        Resize the input image and its corresponding mask to the specified size.

        Args:
        - image (numpy.ndarray): The input image.
        - mask (numpy.ndarray): The corresponding mask image.
        - new_size (tuple): A tuple specifying the new size (width, height).

        Returns:
        - resized_image (numpy.ndarray): The resized image.
        - resized_mask (numpy.ndarray): The resized mask.
        """

        # Resize the image
        resized_image = cv.resize(image, new_size)

        # Resize the mask
        resized_mask = cv.resize(mask, new_size, interpolation=cv.INTER_NEAREST)

        return resized_image, resized_mask[:,:,0]

    def sem_resize(self,gt_sem,pad_size):
        #resize to the largest volume inside the 640x640
                    
        gt_seg = mmcv.imrescale(
            gt_sem,
            pad_size,
            interpolation='nearest',
            backend='cv2')
        # print(f'gt_seg shape {gt_seg.shape}')
        #pad the remaining with background label or ignore label 
        h, w = gt_seg.shape[-2:]
        pad_h, pad_w = pad_size
        gt_seg = F.pad(
            torch.from_numpy(gt_seg),
            pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
            mode='constant',
            value=self.classes_thr-1)
        # print(f'gt_seg shape after pad {gt_seg.shape}')
        return gt_seg
    @staticmethod
    def letterbox_mask(mask, target_size, fill_value=128):
        """
        Letterbox resize the segmentation mask to the target size.

        Parameters:
        - mask: Segmentation mask (numpy array).
        - target_size: Target size (height, width).
        - fill_value: The value used to fill the letterboxed areas.

        Returns:
        - Resized and letterboxed segmentation mask (numpy array).
        """
        h, w = mask.shape[:2]
        target_h, target_w = target_size

        # Calculate the resizing factors while maintaining the aspect ratio
        # aspect_ratio_mask = w / h
        # new_w = int(target_w)
        # new_h = int(target_w / aspect_ratio_mask)
        aspect_ratio_mask = min(target_h/ h, target_w / w)
        new_w = int(target_w)
        new_h = int(target_w / aspect_ratio_mask)


        # Resize the mask using nearest-neighbor interpolation
        resized_mask = cv.resize(mask, (new_w, new_h), interpolation=cv.INTER_NEAREST)
        print(f'mask shaape after first resize {resized_mask.shape}')

        # Create a canvas (target-sized mask) filled with fill_value
        canvas_mask = np.full((target_h, target_w), fill_value, dtype=np.uint8)

        # Calculate the position to paste the resized mask on the canvas
        start_h = (target_h - new_h) // 2
        start_w = (target_w - new_w) // 2

        # Paste the resized mask onto the canvas
        canvas_mask[start_h:start_h + new_h, start_w:start_w + new_w] = resized_mask

        return canvas_mask
    
    @staticmethod
    def resize_and_letter_box(image, rows, cols):
        """
        Letter box (black bars) a color image (think pan & scan movie shown 
        on widescreen) if not same aspect ratio as specified rows and cols. 
        :param image: numpy.ndarray((image_rows, image_cols, channels), dtype=numpy.uint8)
        :param rows: int rows of letter boxed image returned  
        :param cols: int cols of letter boxed image returned
        :return: numpy.ndarray((rows, cols, channels), dtype=numpy.uint8)
        """
        image_rows, image_cols = image.shape[:2]
        row_ratio = rows / float(image_rows)
        col_ratio = cols / float(image_cols)
        ratio = min(row_ratio, col_ratio)
        image_resized = cv.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        letter_box = np.zeros((int(rows), int(cols), 3))
        row_start = int((letter_box.shape[0] - image_resized.shape[0]) / 2)
        col_start = int((letter_box.shape[1] - image_resized.shape[1]) / 2)
        letter_box[row_start:row_start + image_resized.shape[0], col_start:col_start + image_resized.shape[1]] = image_resized
        return letter_box
    





class DownloadSampleDataset(Dataset):
    #constructer
    '''remove the things then download'''
    def __init__(self,transform=None,status="train"):
        #print(config['DATA']['imgpath'])
        self.status=status
        self.img_path=f"/home/jawad/datasets/coco/{self.status}2017" # config['DATA']['imgpath'] #
        self.mask_path=f"/home/jawad/datasets/stuffthingmaps_trainval2017/{self.status}2017" #config['DATA']['maskpath']
        self.img_names_list=sorted_alphanumeric(os.listdir( self.img_path)) # this might be wrong
        self.mask_names_list=sorted_alphanumeric(os.listdir( self.mask_path)) # this might be wrong
        self.num_imgs=len(self.img_names_list)
        # self.chosen_classes=[16,17]
        # self.chosen_classes=[160,127]#stairs house
        self.chosen_classes=[112]
        # self.chosen_classes=list(range(91,91+91)) #[105,168]

     
    #len 
    def __len__(self):
        return  self.num_imgs

    #getitem
    def __getitem__(self, index):
        #read image 
        img=cv.imread(os.path.join(self.img_path,self.img_names_list[index]))
        #print(img.shape)
        #read label image
        mask=cv.imread(os.path.join(self.mask_path,self.mask_names_list[index]),cv.IMREAD_GRAYSCALE) 
        unique_list=np.unique(mask).tolist()
  
        temp=[i for i in self.chosen_classes if i in unique_list]
        mask_new=np.ones_like(mask)*len(self.chosen_classes) #TODO: this will not work if classs equal to len
        if len(temp)>0:
            #set all other classes to len(self.chosen_classes)
            for id,c in enumerate(self.chosen_classes):
                mask_new[mask==c]=id
            
            #zprint(mask)
            print(np.unique(mask_new))
            #create the folders automatically
            ##if true save the mask and img
            cv.imwrite(rf'/home/jawad/datasets/{self.status}sample2017/images/{self.img_names_list[index]}',img)  #not consistent
            cv.imwrite(rf'/home/jawad/datasets/{self.status}sample2017/semantic_labels/{self.mask_names_list[index]}',mask_new)    


        return img
    
class DownloadSampleDatasetOther(Dataset):
    #constructer
    '''remove the things then download'''
    def __init__(self,transform=None,status="train"):
        #print(config['DATA']['imgpath'])
        self.status=status
        self.img_path=f"/home/jawad/datasets/coco/{self.status}2017" # config['DATA']['imgpath'] #
        self.mask_path=f"/home/jawad/datasets/stuffthingmaps_trainval2017/{self.status}2017" #config['DATA']['maskpath']
        self.img_names_list=sorted_alphanumeric(os.listdir( self.img_path)) # this might be wrong
        self.mask_names_list=sorted_alphanumeric(os.listdir( self.mask_path)) # this might be wrong
        self.num_imgs=len(self.img_names_list)
        # self.chosen_classes=[16,17]
        # self.chosen_classes=[160,127]#stairs house
        self.chosen_classes=[112]
        # self.chosen_classes=list(range(91,91+91)) #[105,168]

     
    #len 
    def __len__(self):
        return  self.num_imgs

    #getitem
    def __getitem__(self, index):
        #read image 
        img=cv.imread(os.path.join(self.img_path,self.img_names_list[index]))
        #print(img.shape)
        #read label image
        mask=cv.imread(os.path.join(self.mask_path,self.mask_names_list[index]),cv.IMREAD_GRAYSCALE) 
        unique_list=np.unique(mask).tolist()
        # print(unique_list)
        temp=[i for i in self.chosen_classes if i in unique_list]
        mask_new=np.ones_like(mask)*len(self.chosen_classes) #TODO: this is for other
        mask_new[mask==255]=len(self.chosen_classes) + 1 #This is a label for background
        if len(temp)>0:
            #set all other classes to len(self.chosen_classes)
            for id,c in enumerate(self.chosen_classes):
                mask_new[mask==c]=id
            
            #zprint(mask)
            print(np.unique(mask_new))
            #create the folders automatically
            ##if true save the mask and img
            cv.imwrite(rf'/home/jawad/datasets/{self.status}sample2017/images/{self.img_names_list[index]}',img)  #not consistent
            cv.imwrite(rf'/home/jawad/datasets/{self.status}sample2017/semantic_labels/{self.mask_names_list[index]}',mask_new)    


        return img
    
    

if __name__=="__main__":
    import matplotlib.pyplot as plt

    def visualize_image_and_mask(image, mask):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the image
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Display the mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        plt.show()
    # L=LetterResize((640,640),use_mini_pad=False,stretch_only=False,allow_scale_up=False,half_pad_param=False)
    dataset=DownloadSampleDatasetOther(status="train")
    dataloader=DataLoader(dataset,batch_size=1,shuffle=
                          False)


    for id,data in enumerate(dataloader):

        print(id)
        
       
    # num_classes=91 + 1 #including background

    # # Initialize your segmentation dataset (replace with your dataset class)
    # dataset = SemanticDataset(classes_threshold=num_classes,status="train") #not plus one since starts at 0

    # # Set up data loader
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True) #TODO: check num of workers

    # for data in data_loader:
    #     img,mask=data
    #     print(1)
    #     visualize_image_and_mask(img[0], mask[0])
