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


class Identity(nn.Module):
    def forward(self, x):
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size):
        super(FeatureExtractor, self).__init__()

        self.embedding_size = embedding_size
        self.resnet = models.alexnet(pretrained=True)
        self.resnet.classifier = Identity()

        self.dropout_0 = nn.Dropout(p=0.5)
        self.fc_0 = nn.Linear(256 * 6 * 6, self.embedding_size, bias=True)

        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout_0(x)
        h = self.fc_0(x)
        x = self.relu_1(h)
        x = self.dropout_1(x)
        x = self.fc_1(x)
        return x, h


class SiameseNet(nn.Module):
    def __init__(self, num_persons):
        super(SiameseNet, self).__init__()

        self.embedding_size = 128
        self.feature_extractor = FeatureExtractor(self.embedding_size)
        self.classifier_layer = nn.Linear(self.embedding_size, num_persons)

        self.dist = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()
        self.lin = nn.Linear(1, 1, bias=True)
        self.lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(1)

    def forward(self, x):
        a, _ = self.feature_extractor(x[0])
        b, _ = self.feature_extractor(x[1])

        dist = self.dist(a, b)
        dist = self.sigmoid(self.lin(dist.unsqueeze(1))).squeeze()

        id_0 = self.classifier_layer(a)
        id_1 = self.classifier_layer(b)

        return dist, id_0, id_1

    def extract_features(self, x):
        # use this code to extract the feature of a single image
        _, h = self.feature_extractor(x)
        return h


preprocess_reid_image = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])


def main():
    net = SiameseNet(5898).to(device)
    print("device: " + device.__str__())
    net.load_state_dict(torch.load('./trained_nets/april_2020_v9_2.pyt', map_location=torch.device('cpu')))

    img_1 = Image.open('./images/1.jpg')
    img_1 = preprocess_reid_image(img_1)
    img_1 = img_1.unsqueeze(0)

    img_2 = Image.open('./images/2.jpg')
    img_2 = preprocess_reid_image(img_2)
    img_2 = img_2.unsqueeze(0)

    features = torch.zeros(2, 128)

    with torch.no_grad():
        net.eval()
        features[0] = net.extract_features(img_1.to(device)).cpu()
        features[1] = net.extract_features(img_2.to(device)).cpu()

    np.savetxt('features.csv', features, delimiter=',')

    # similarity must be calculated using cosine distance
    # similarity scores are therefore between -1 and 1

    similarity_score = F.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
    print('Similarity score different images', str(similarity_score.item()))

    similarity_score = F.cosine_similarity(features[1].unsqueeze(0), features[1].unsqueeze(0))
    print('Similarity score same images', str(similarity_score.item()))


if __name__ == "__main__":
    main()
