import torch
import os
import pdb
import torch.nn as nn
from PIL import Image
from PIL import ImageFilter
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler
import numpy as np
import torch.nn.functional as F
import argparse
import math

# set a fixed random seed
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prepare_reid_image = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                        ])

class Identity(nn.Module):
    def forward(self,x):
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size):
        super(FeatureExtractor, self).__init__()

        self.embedding_size = embedding_size
        self.resnet = models.alexnet(pretrained=True)
        self.resnet.classifier = Identity()
        
        self.dropout_0 = nn.Dropout(p=0.5)
        self.fc_0 = nn.Linear(256 * 6 * 6,self.embedding_size,bias = True)            

        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(self.embedding_size,self.embedding_size,bias = True)            

    def forward(self, x):        
        x = self.resnet(x)               
        x = self.dropout_0(x)        
        h = self.fc_0(x)        
        x = self.relu_1(h)
        x = self.dropout_1(x)
        x = self.fc_1(x)           
        return x,h

class SiameseNet(nn.Module):
    def __init__(self, num_persons):
        super(SiameseNet, self).__init__()
        
        self.embedding_size = 128
        self.feature_extractor = FeatureExtractor(self.embedding_size)  
        self.classifier_layer = nn.Linear(self.embedding_size,num_persons)   

        self.dist = nn.CosineSimilarity()  
        self.sigmoid = nn.Sigmoid()
        self.lin = nn.Linear(1,1, bias=True)
        self.lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(1)

    def forward(self, x):

        a,_ = self.feature_extractor(x[0])
        b,_ = self.feature_extractor(x[1])

        dist = self.dist(a,b)
        #dist = self.sigmoid(dist)
        dist = self.sigmoid(self.lin(dist.unsqueeze(1))).squeeze()

        id_0 = self.classifier_layer(a)
        id_1 = self.classifier_layer(b)

        return dist, id_0, id_1

    def extract_features(self,x):
        # use this code to extract the feature of a single image
        _,h = self.feature_extractor(x)
        return h

def list_files(dataset_dir):
    files_list = []
    for f in os.listdir(dataset_dir):
        if f.endswith('.jpg') and os.path.isfile(os.path.join(dataset_dir,f)):
            files_list.append(f)
    return files_list

net = SiameseNet(5898).to(device)
net.load_state_dict(torch.load('./trained_nets/april_2020_v9_2.pyt',map_location=torch.device('cpu')))
net.eval()

dataset_dirs = ['armed_police','police_high_vis','staff_blue_jacket','staff_high_vis','general_public']

for d in range(len(dataset_dirs)):
    dataset_dir = './/staff_dataset//' + dataset_dirs[d]
    output_dir = dataset_dir

    image_files = list_files(dataset_dir)
    features = torch.zeros(len(image_files),128)

    for i,f in enumerate(image_files):
        print(f)
        img_1 = Image.open(os.path.join(dataset_dir,f))   
        img_1 = prepare_reid_image(img_1)
        img_1 = img_1.unsqueeze(0)
        with torch.no_grad():
            net.eval()
            features[i] = net.extract_features(img_1.to(device)).cpu()

    np.savetxt(os.path.join(output_dir,'features.csv'), features, delimiter=',')
