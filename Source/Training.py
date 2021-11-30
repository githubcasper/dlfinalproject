# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:07:56 2021

@author: Andreas Tind
"""

from Dataloader import NewsDataset
import neptune.new as neptune
import keyring
#import os
#from pathlib import Path as PL


#%% Neptune

'''
secret_api = keyring.get_password('Neptune', "andreastind")

run = neptune.init(project="andreastind/DeepLearningFinalProject",
                   api_token=secret_api)

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66
run.stop()
'''

#%% Training

json_path = "..\\Data\\News_Category_Dataset_v2.json"

tmp = NewsDataset(json_path)




