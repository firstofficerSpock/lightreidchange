import argparse, ast
import sys
sys.path.append('../..')
sys.path.append('.')
import torch
import lightreid
import yaml
import time
import numpy as np

import cv2
import numpy as np
from torchvision import transforms
#from somewhere.inference import Inference  # 确保正确导入 Inference 类

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='/home/vipuser/Desktop/light-reid/examples/bagtricks_buildwithconfigs/base_config_duke_res18.yaml', help='')
parser.add_argument('--model_path', type=str, default='/home/vipuser/Desktop/light-reid/examples/bagtricks_buildwithconfigs/model_120.pth', help='')
args = parser.parse_args()

# load configs from yaml
with open(args.config_file) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# inference only
inference = lightreid.build_inference(config, model_path=args.model_path, use_gpu=True)

# 打开视频文件
video = cv2.VideoCapture('/home/vipuser/Desktop/light-reid/examples/bagtricks_buildwithconfigs/MOT16-03.mp4')
if not video.isOpened():
    raise Exception("Could not open video")

# 视频帧预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # 假设我们要将每帧调整为 224x224 大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 存储所有特征的列表
all_features = []

while True:
    ret, frame = video.read()
    if not ret:
        break  # 如果没有帧了，就结束循环

    # 将 BGR 图像帧转换为 RGB 格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 应用预处理转换
    frame_tensor = preprocess(frame_rgb)

    # 转换前确保 frame_tensor 的形状是 (C, H, W)
    if frame_tensor.ndim == 3:
        frame_tensor = frame_tensor.unsqueeze(0)  # 添加一个批次维度

    # 将帧的张量转换为 numpy 数组
    frame_numpy = frame_tensor.numpy()

    # 直接传递 numpy 数组到 inference.process 方法
    features = inference.process(frame_numpy, return_type='numpy')  # 此处不使用列表
    all_features.append(features)

    
# 计算特征之间的相似度矩阵
all_features = np.concatenate(all_features, axis=0)
similarity_matrix = np.matmul(all_features, all_features.transpose(1, 0))

# 打印余弦相似度矩阵的形状和矩阵本身
print(f'Cosine similarity matrix shape: {similarity_matrix.shape}')
print(similarity_matrix)

np.save('dim512.npy', all_features)  # 可以替换 'unique_features_name.npy' 为任何您想要的文件名

# process
#img_paths = [
#    './imgs/3006_c1s1_f000.jpg',
#    './imgs/3006_c2s1_f000.jpg',
#    './imgs/3007_c1s1_f000.jpg',
#    './imgs/3007_c2s1_f000.jpg',
#    './imgs/3008_c1s1_f000.jpg',
#    './imgs/3008_c2s1_f000.jpg',
#    './imgs/3013_c1s1_f000.jpg',
#    './imgs/3013_c2s1_f000.jpg',
#]
#features = inference.process(frame_tensor, return_type='numpy')
#all_features.append(features)
# compute distance
#print('feature shape: {}'.format(features.shape))
#cosine_similarity = np.matmul(features, features.transpose([1,0])) # inner-produce distance (i.e. cosine distance since the feature has been normalized)
#print('cosine similarity as below: ')
#print(cosine_similarity)

#else:
#    print("Frame is empty or has zero size.")
#else:
#    print("Skipped empty frame or video has ended.")
#    continue
video.release()        

