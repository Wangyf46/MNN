# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import MNN
import cv2
import pdb
import os
import json
import numpy as np
from skimage import exposure
from termcolor import cprint


def log_print(text, log_file, color = None, on_color = None, attrs = None):
    print(text, file=log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)


class speedlimit(object):
    def __init__(self):
        self.img_file = '/data/workspace/mixed-data/Images/'
        self.ann_file = '/data/workspace/mixed-data/test.json'

        self.name_list = os.listdir(self.img_file)

        # with open(self.ann_file, 'r') as f:
        #     data = json.load(f)
        #     self.images = data['images']
        #     self.annotations = data['annotations']

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        path = self.img_file + self.name_list[idx]
        image = cv2.imread(path)
        return image


def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter('/home/wyf/codes/traffic-sign-classification/models/LeNet5_18.mnn')
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    data = speedlimit()
    TP = np.zeros(18)
    T = np.zeros(18)
    record_file = open('/home/wyf/codes/MNN/pymnn/examples/MNNEngineDemo/shufflenet_v2_ssd_480x256.txt', 'w')

    for i_batch, data in enumerate(data):
        img = data[..., ::-1]
        img = cv2.resize(img, (480, 256))
        img = img.astype(float)

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose((2, 0, 1))

        tmp_img = MNN.Tensor((1, 3, 256, 480), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe)

        #construct tensor from np.ndarray
        pdb.set_trace()
        input_tensor.copyFrom(tmp_img)
        interpreter.runSession(session)
        output_tensor = interpreter.getSessionOutput(session)
        print("output belong to class: {}".format(np.argmax(output_tensor.getData())))
        pdb.set_trace()

    #     pred = np.argmax(output_tensor.getData())
    #     log_print("output belong to class: {}; Label: {}".format(pred, blob['label']), record_file)
    #
    #     if blob['label'] == pred:
    #         TP[pred] += 1
    #     T[blob['label']] += 1
    # RECALL = TP / T
    # avg_acc = np.sum(TP) / np.sum(T)
    #
    # with open('/data/workspace/speed-limit/speedlimit.label', 'r') as fcat:
    #     cats = fcat.readlines()
    #     cats = list(map(lambda x: x.strip(), cats))
    #
    # log_print('***************dist: ****************************************', record_file)
    # log_print('{:<8}{:<15}{:<15}{:<15}{:<15}'.format('idx', 'category', 'TP', 'T', 'RECALL'), record_file)
    # log_print('-------------------------------------------------------------', record_file)
    # for idx in range(18):
    #     log_print('{:<8}{:<15}{:<15}{:<15}{:<5}'.format(str([idx]), cats[idx], TP[idx], T[idx], RECALL[idx]), record_file)
    # log_print('-------------------------------------------------------------', record_file)
    # log_print('{:<8}{:<15}{:<15}{:<15}{:<15}'.format('', 'total', np.sum(TP), np.sum(T), avg_acc), record_file)


if __name__ == "__main__":
    inference()
