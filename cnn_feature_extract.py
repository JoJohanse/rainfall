import torch
import torchvision as tv
import cv2
import os
import numpy as np



class cnn_feature_extract:
    '''
    cnn模型建模
    '''
    def __init__(self):
        device = torch.device("cuda:0")
        self.__net=torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=2),  # 增加padding以增加宽度和高度
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),  # 减小stride以增加宽度和高度 ps:硬凑形状
        ).to(device)

        self.__net.eval()


        self.__final_pool=torch.nn.MaxPool2d(3,2)
        

    '''
    其他的与原resnet模型完全一致,调用方法也一致
    '''
    def __img_pr(self,img):
        #图片预处理

        transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
            ])
        img=transform(img)
        img=torch.autograd.Variable(
            torch.unsqueeze(img,dim=0),requires_grad=True
            )
        return img

    def __getfeature(self,input):
        device = torch.device("cuda:0")
        n=self.__net
        pool=self.__final_pool
        input=input.to(device)
        with torch.no_grad():
            x=n(input)
            x=pool(x)
        return x

    def GetFeature(self,imgPath):
        img=cv2.imread(imgPath)
        if img is None:
            raise Exception('Load image@{} failed'.format(imgPath))
        img=self.__img_pr(img)
        return self.__getfeature(img)

    def GetFeature2(self,img):
        img=self.__img_pr(img)
        return self.__getfeature(img)

if __name__=='__main__':
    path=os.path.dirname(__file__)
    img_path=path+os.sep+'o (1).jpg'
    img=cv2.imread(img_path)
    if img is None:
        raise Exception('image')
    img=cnn_feature_extract._img_pr(img)
    print(cnn_feature_extract.getfeature(img).size())
