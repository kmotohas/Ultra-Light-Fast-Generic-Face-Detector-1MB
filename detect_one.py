"""
This code is used to batch detect images in a folder.
"""
#import argparse
import os
import sys

import cv2
import base64
import json

from vision.ssd.config.fd_config import define_img_size

#parser = argparse.ArgumentParser(
#    description='detect_imgs')
#
#parser.add_argument('--net_type', default="mb_tiny_RFB_fd", type=str,
#                    help='The network architecture ,optional:1. mb_tiny_RFB_fd (higher precision) or 2.mb_tiny_fd (faster)')
#parser.add_argument('--input_size', default=640, type=int,
#                    help='define network input size,default optional value 128/160/320/480/640/1280')
#parser.add_argument('--threshold', default=0.7, type=float,
#                    help='score threshold')
#parser.add_argument('--candidate_size', default=1500, type=int,
#                    help='nms candidate size')
#parser.add_argument('--path', default="imgs", type=str,
#                    help='imgs dir')
#parser.add_argument('--test_device', default="cpu", type=str,
#                    help='cuda:0 or cpu')
#args = parser.parse_args()
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

#if not os.path.exists(result_path):
#    os.makedirs(result_path)
#listdir = os.listdir(path)
#sum = 0
#for file_path in listdir:
#img_path = os.path.join(args.path, file_path)
#orig_image = cv2.imread(img_path)
#image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
def Base64ToNdarray(img_base64):
    img_data = base64.b64decode(img_base64)
    img_np = np.fromstring(img_data, np.uint8)
    image = cv2.imdecode(img_np, cv2.COLOR_BGR2RGB)
    return image
#dec = json.loads(image[0])
with open('/home/ubuntu/test.txt', mode='w') as f:
    f.write(image)
image = eval(image)
"""
image = image['default']
#image = np.squeeze(image)
image = Base64ToNdarray(image)
boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
print(type(boxes))
print(len(boxes))

nboxes = json.dumps({"default":'+ str(len(boxes))+'})
"""
nboxes = json.dums({"default": "test"})
#sum += boxes.size(0)
#for i in range(boxes.size(0)):
    #box = boxes[i, :]
    #cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    #label = f"{probs[i]:.2f}"
    # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#cv2.imwrite(os.path.join(result_path, file_path), orig_image)
#print(f"Found {len(probs)} faces. The output image is {result_path}")
#print(sum)
