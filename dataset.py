import os
from torchvision.io import read_image
import torch
import pandas as pd
from http.client import FOUND
from torch.utils.data import Dataset, DataLoader

# from deepface import DeepFace
# import cv2
import random
random.seed(10)
import os
from skimage import io
import regex as re
import json
import pandas as pd
import shutil

class RFW_CB_Dataset(Dataset):
    base = "data/txts"
    data = ""
    paths = list()
    label = dict()

    def __init__(self, csv_file, root_dir, transform=None):
        self.faces = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.faces.iloc[idx, 0])
        img1_name = os.path.join(self.root_dir, eval(self.faces.iloc[idx, 1])[0])
        img2_name = os.path.join(self.root_dir, eval(self.faces.iloc[idx, 1])[1])

        images = [io.imread(img1_name), io.imread(img2_name)]
        
        label = self.faces.iloc[idx, 2]
        label = label.astype('float')

        sample = {'image': images, 'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample
            
def get_labels(race:str, ratio:float):
    # takes the ratio to divide  training/tst set
    # then iterate the data with people of race "race" then return the train/test set and labels 
        data = "data/txts/" + race
        print('start')

        path1 = list()
        path2 = list()
        labels = list()

        # df = pd.DataFrame({key:[] for key in ["path1", "path2", "label"]})
        with open(data+"/"+race+"_pairs.txt", "r") as f:
            # count = 0
            for row in f.readlines():
                # count += 1
                # if count == 10:
                #     break
                row = row.replace("\n", "").split("\t")
                label = 1
                id1 = ""
                id2 = ""
                img1 = ""
                img2 = ""

                if len(row) == 3:
                    # the same person
                    img1 = row[0]
                    img2 = img1
                    id1 = row[1]
                    id2 = row[2]

                else:
                    label = 0
                    img1 = row[0]
                    img2 = row[2]
                    id1 = row[1]
                    id2 = row[3]    
                
                path1.append("data/" + race + "/" + img1 + "/" + img1+"_000"  + str(id1) + ".jpg")
                path2.append("data/" + race + "/" + img2 + "/" + img2+"_000" + str(id2) + ".jpg")
                labels.append(label)

        path_pair = list(zip(path1, path2))
        
        label_di = dict(zip(list(path_pair), labels))

        df = pd.DataFrame({key: [] for key in ["id", "label"]})
        df["id"] = path_pair
        df['label'] = labels

        df.to_csv(race + "_dataset.csv")



if __name__=="__main__":

  
    race = "Asian"
    get_labels("Asian", 0.8)    

    dl = RFW_CB_Dataset(race + "_dataset.csv", ".")
    dataloader = DataLoader(dl, batch_size=16)

    for item in dataloader:
        print(item)
        


    # validify data partitioning
    # for key, val in label.items():
    #     print('a')
    #     assert(key[0].split("/")[2] == key[1].split("/")[2]) == val

    