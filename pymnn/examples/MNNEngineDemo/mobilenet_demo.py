# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import MNN
import cv2
import pdb
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


def preprocess(img_path, category_id=None, aug=False):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (32, 32))
    img = (img / 255.).astype(np.float32)
    img = exposure.equalize_adapthist(img)

    # Convert to one-hot encoding
    # if category_id is not None:
    #     label = np.eye(len(CLASS))[category_id]

    img = img.reshape(img.shape + (1,)).transpose((2,0,1)).astype(np.float32)

    # if aug:
    #     img = extra_aug(img)

    return img, category_id

class speedlimit(object):
    def __init__(self):
        # self.img_file = '/data/workspace/speed-limit/Non-Negative/test' + '.txt'
        # self.ann_file = '/data/workspace/speed-limit/Non-Negative/test' + '.json'

        self.img_file = '/data/workspace/speed-limit/test' + '.txt'
        self.ann_file = '/data/workspace/speed-limit/test' + '.json'

        with open(self.img_file, 'r') as fsets:
            self.name_list = list(map(lambda x: x.strip(), fsets.readlines()))

        with open(self.ann_file, 'r') as f:
            data = json.load(f)
            self.images = data['images']
            self.annotations = data['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]['file_name']
        category_id = self.annotations[idx]['category_id']
        image_id = self.annotations[idx]['image_id']
        img, label = preprocess(fname, category_id)
        blob = {}
        blob['image'] = img
        blob['fname'] = fname
        blob['label'] = category_id
        blob['image_id'] = image_id
        return blob


def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter('/home/wyf/codes/traffic-sign-classification/models/LeNet5_18.mnn')
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    data = speedlimit()
    TP = np.zeros(18)
    T = np.zeros(18)
    record_file = open('/home/wyf/codes/MNN/pymnn/examples/MNNEngineDemo/LeNet5_18.txt', 'w')

    for i_batch, blob in enumerate(data):
        img = blob['image']
        tmp_img = MNN.Tensor((1, 1, 32, 32), MNN.Halide_Type_Float, img, MNN.Tensor_DimensionType_Caffe)

        #construct tensor from np.ndarray
        input_tensor.copyFrom(tmp_img)
        interpreter.runSession(session)
        output_tensor = interpreter.getSessionOutput(session)
        pred = np.argmax(output_tensor.getData())
        log_print("output belong to class: {}; Label: {}".format(pred, blob['label']), record_file)
        
        if blob['label'] == pred:
            TP[pred] += 1
        T[blob['label']] += 1
    RECALL = TP / T
    avg_acc = np.sum(TP) / np.sum(T)

    with open('/data/workspace/speed-limit/speedlimit.label', 'r') as fcat:
        cats = fcat.readlines()
        cats = list(map(lambda x: x.strip(), cats))

    log_print('***************dist: ****************************************', record_file)
    log_print('{:<8}{:<15}{:<15}{:<15}{:<15}'.format('idx', 'category', 'TP', 'T', 'RECALL'), record_file)
    log_print('-------------------------------------------------------------', record_file)
    for idx in range(18):
        log_print('{:<8}{:<15}{:<15}{:<15}{:<5}'.format(str([idx]), cats[idx], TP[idx], T[idx], RECALL[idx]), record_file)
    log_print('-------------------------------------------------------------', record_file)
    log_print('{:<8}{:<15}{:<15}{:<15}{:<15}'.format('', 'total', np.sum(TP), np.sum(T), avg_acc), record_file)

'''
def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter('/home/wyf/codes/traffic-sign-classification/LeNet5.mnn')
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    image = cv2.imread('ILSVRC2012_val_00049999.JPEG')
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224))
    #resize to mobile_net tensor size
    image = image.astype(float)
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #preprocess it
    image = image.transpose((2, 0, 1))
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(output_tensor.getData())))
'''
if __name__ == "__main__":
    inference()
