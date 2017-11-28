import os
import time
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


path = "/home/wonjun/Desktop/pycharm-2017.2.3//PycharmProjects/RCNN_BCI/"
RCNN_summary_path = create_dir(path + "summary_RCNN/%d/"%(time.time()))
RCNN_model_path = create_dir(path + "model_RCNN/Experiment_10/sbj13789/")
RCNN_data_path = path + "data/"
RCNN_uncued_data_path = path + "data/BCI_Competition_IV_IIa/Uncued_data/"
RCNN_separated_class = path + "data/BCI_Competition_IV_IIa/separated_class/"
Topography_path = create_dir(path + "Figs/Feature_map/")


channel_cnt = 22
time_cnt = 512  #2sec
batch_size = 64
num_feature_map = 64