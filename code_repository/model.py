import json
import os
from torchvision.models import *
import wandb
from sklearn.model_selection import train_test_split
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import *
import torch,torchvision
from tqdm import tqdm
device = 'cpu'
PROJECT_NAME = 'Animal-Clf'

class DataLoad:
    
    def load_data():

        labels_l = {}
        labels_r = {}
        idx = 0
        data = []
        for folder in os.listdir('/content/drive/MyDrive/Find my lover/animal/animals/animals'):
        if folder in labels:
            idx += 1
            labels_l[folder] = idx
            labels_r[idx] = folder
        for folder in tqdm(os.listdir('/content/drive/MyDrive/Find my lover/animal/animals/animals')):
            if folder in labels:
            for file in os.listdir(f'/content/drive/MyDrive/Find my lover/animal/animals/animals/{folder}/'):
                img = cv2.imread(f'/content/drive/MyDrive/Find my lover/animal/animals/animals/{folder}/{file}')
                img = cv2.resize(img,(56,56))
                img = img / 255.0
                data.append([
                    img,
                    np.eye(
                        labels_l[folder]+1,len(labels_l)
                    )[labels_l[folder]]
                ])

        X = []
        y = []
        for d in data:
            X.append(d[0])
            y.append(d[1])
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,shuffle=False)
        X_train = torch.from_numpy(np.array(X_train)).to(device).view(-1,3,56,56).float()
        y_train = torch.from_numpy(np.array(y_train)).to(device).float()
        X_test = torch.from_numpy(np.array(X_test)).to(device).view(-1,3,56,56).float()
        y_test = torch.from_numpy(np.array(y_test)).to(device).float()
        return X,y,X_train,X_test,y_train,y_test,labels_l,labels_r,idx,data


    def load_men_data():

        img_list=[]
        data=[]    
        folder_path='/content/drive/MyDrive/Find my lover/women/'
        for i in range(217):
        
        img_path=str(i+1)+".jpg"
        try:
            img = cv2.imread(folder_path+img_path)
            img_t = cv2.resize(img,(56,56))
            img_t = img_t / 255.0
            data.append(img_t)

            img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        except: continue
                        
        men= torch.from_numpy(np.array(data)).to(device).view(-1,3,56,56).float()
        
        return men, img_list



class Model(Module):

    def __init__(self):
        
        super().__init__()
        self.max_pool2d = MaxPool2d((2,2),(2,2))
        self.activation = ReLU()
        self.conv1 = Conv2d(3,7,(5,5))
        self.conv1bn = BatchNorm2d(7)
        self.conv2 = Conv2d(7,14,(5,5))
        self.conv2bn = BatchNorm2d(14)
        self.conv3 = Conv2d(14,21,(5,5))
        self.conv3bn = BatchNorm2d(21)
        self.linear1 = Linear(21*3*3,256)
        self.linear1bn = BatchNorm1d(256)
        self.linear2 = Linear(256,512)
        self.linear2bn = BatchNorm1d(512)
        self.linear3 = Linear(512,256)
        self.linear3bn = BatchNorm1d(256)
        self.output = Linear(256,len(labels))
    
    def forward(self,X):
        preds = self.max_pool2d(self.activation(self.conv1bn(self.conv1(X))))
        preds = self.max_pool2d(self.activation(self.conv2bn(self.conv2(preds))))
        preds = self.max_pool2d(self.activation(self.conv3bn(self.conv3(preds))))
        preds = preds.view(-1,21*3*3)
        preds = self.activation(self.linear1bn(self.linear1(preds)))
        preds = self.activation(self.linear2bn(self.linear2(preds)))
        preds = self.activation(self.linear3bn(self.linear3(preds)))
        preds = self.output(preds)
        return preds

    def get_loss(model,X,y,criterion):
        preds = model(X)
        loss = criterion(preds,y)
        return loss.item()

    def get_accuracy(model,X,y):
        correct = 0
        total = 0
        preds = model(X)
        for pred,y_batch in zip(preds,y):
            pred = int(torch.argmax(pred))
            y_batch = int(torch.argmax(y_batch))
            if pred == y_batch:
                correct += 1
            total += 1
        acc = round(correct/total,3)*100
        return acc

    def get_animal_type(model, X):
        pred = model(X)
        pred = int(torch.argmax(pred))
        return pred