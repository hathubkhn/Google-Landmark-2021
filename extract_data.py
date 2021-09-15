import os
import pandas as pd
import shutil
import sys
import os

init_path = os.path.abspath("./")
if not os.path.exists("train_data"): os.mkdir("train_data")
def extract_1000_biggest_class():
    csv_data = pd.read_csv(os.path.join(init_path, "train.csv"), sep = ",")
    keys = csv_data["landmark_id"].value_counts().keys().tolist()
    values = csv_data["landmark_id"].value_counts().tolist()
    dict_ = dict(zip(keys, values))
    
    dict_  = {k:dict_[k] for k in list(dict_.keys())[:1000]}
    frames = []
    
    for k in dict_.keys():
        indexes = csv_data.index[csv_data["landmark_id"]==k].tolist()
        frames.append(csv_data.loc[indexes,:])
        list_id = list(csv_data.loc[indexes, "id"])
        print(f"[COPY] class {k}")
        for id in list_id:
            input_path = f"train/{id[0]}/{id[1]}/{id[2]}"
            shutil.copy(os.path.join(input_path, f"{id}.jpg"), f"train_data/{id}.jpg")
    
    df = pd.concat(frames, axis = 0)
    headers =  ["id", "label"]
    df.columns = headers
    df.to_csv(os.path.join(init_path,"train_new.csv"), sep = "\t", index = False)

def change_label(train_file):
    df = pd.read_csv(train_file, sep = "\t")
    list_ = list(df["label"])
    list_1 = list(set(list_))
    print(list_1[:10])
    dict_ = {v:k for k, v in enumerate(list_1)}
    list_new = [dict_[i] for i in list_]
    df["new_label"] = list_new
    df = df.drop("label", axis = 1)
    df.to_csv(os.path.exists(init_path,"train_new.csv"), sep = "\t", index = False)

if __name__ == '__main__':
    input_file = os.path.join(init_path, "train_new.csv")
    change_label(input_file)
    
