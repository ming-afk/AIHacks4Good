import os
from torchvision.io import read_image
import torch
import pandas as pd
from http.client import FOUND
from torch.utils.data import Dataset

# from deepface import DeepFace
# import cv2
import random
random.seed(10)
import os
import regex as re
import json
import pandas as pd
import shutil

class RFW_CB_Dataset(Dataset):
    base = "data/txts"
    data = ""
    paths = list()
    label = dict()

    def __init__(self, paths, labels):
        self.label= labels
        self.paths = paths

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

        # print("extracting finished")
        # df['path1'] = path1
        # df['path2'] = path2
        # df['label'] = labels      

        # rarr = list()
        # for i in range(len(df)):
        #     rarr.append(random.uniform(0,1))
        
        # trarr = rarr[:int(len(df) * ratio)]
        # while len(trarr) < len(df):
        #     trarr.append(0)

        # while len(tstarr) < len(df):
        #     tstarr.append(0)

        # tstarr = rarr[int(len(df)* ratio):]


        path_pair = list(zip(path1, path2))
        
        label_di = dict(zip(list(path_pair), labels))

        return (path_pair, label_di)



if __name__=="__main__":
    # img = load_image("image/IMG_0084.jpg")
	# base_img = img.copy()

    # img = cv2.imread(os.path.expanduser("~/Downloads/deepface/image/pexels-photo-2379005.jpeg"))
    # cv2.namedWindow("Images")
    # cv2.imshow('Images',img)
    # cv2.waitKey(1)

    # false_poitive()/Users/minghaoli/Downloads/fairface-img-margin025-trainval/train

    # res = DeepFace.analyze("/Users/minghaoli/Downloads/deepface/Face_Recognition/UTKFace/96_1_1_20170110183853718.jpg.chip.jpg", enforce_detection=False)
    # print(type(res))
    # print(res)

    # verify()
    # analyze()
    # partition_gender_race()
    race = "Asian"
    (path_pair, label) = get_labels("Asian", 0.8)    
   
    df = pd.DataFrame({key: [] for key in ["id", "label"]})
    df["id"] = path_pair
    df['label'] = label

    df.to_csv(race + "_dataset.csv")

    # for key, val in label.items():
    #     print('a')
    #     assert(key[0].split("/")[2] == key[1].split("/")[2]) == val

