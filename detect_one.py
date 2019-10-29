import os
import sys

import cv2
import base64
import json

from vision.ssd.config.fd_config import define_img_size

net_type = 'mb_tiny_RFB_fd'
input_size = 640
threshold = 0.7
candidate_size = 1500
path = 'imgs'
test_device = 'cpu'
define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

workdir = '/home/ubuntu/work/Ultra-Light-Fast-Generic-Face-Detector-1MB/'
result_path = workdir + "detect_imgs_results"
label_path = workdir + "models/voc-model-labels.txt"
test_device = test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if net_type == 'mb_tiny_fd':
    model_path = workdir + "models/pretrained/Mb_Tiny_FD_train_input_320.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'mb_tiny_RFB_fd':
    model_path = workdir + "models/pretrained/Mb_Tiny_RFB_FD_train_input_320.pth"
    # model_path = "models/pretrained/Mb_Tiny_RFB_FD_train_input_640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

def Base64ToNdarray(img_base64):
    img_data = base64.b64decode(img_base64)
    img_np = np.fromstring(img_data, np.uint8)
    image = cv2.imdecode(img_np, cv2.COLOR_BGR2RGB)
    return image

image = default  # sent from httpclient
image = Base64ToNdarray(image)
boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)

default = str(len(boxes))

