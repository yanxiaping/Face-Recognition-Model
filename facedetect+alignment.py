import math
import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
# from matplotlib import patches


def face_detect(path):
    mtcnn = MTCNN(margin=5,device='cuda:0',keep_all=False,selection_method='probability')
    img = plt.imread(path)
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return img,probs,landmarks
    boxes = boxes[0]
    if boxes is not None and len(boxes) > 1:
        a = 0
    else:
        a = 1

    h,w = img.shape[:2]
    bx1 = max(0,boxes[0])
    by1 = max(0,boxes[1])
    bw = min(w,boxes[2])-bx1
    bh = min(h,boxes[3]) - by1
    # 裁剪图片
    crop_image = np.array(img)[int(by1):int(by1+bh),int(bx1):int(bx1+bw)]

    if a==0:
        landmarks = landmarks[0] - [bx1, by1]
    else:
        landmarks = np.squeeze(landmarks - [bx1, by1])
    return crop_image,probs,landmarks


def Alignment(crop_img,landmark,output_size = (224,224),eye_distance = 80):
    if landmark is None:
        return None
    x = landmark[1,0]- landmark[0,0]
    y = landmark[1,1]-landmark[0,1]
    eye_center = ((landmark[0,0] + landmark[1,0]) // 2,  # 眼睛中心点
                  (landmark[0,1] + landmark[1,1]) // 2)
    if x == 0:
        angel = 0
    else:
        angel = math.atan(y/x)*180/math.pi

    current_eye_dist = np.sqrt(x ** 2 + y ** 2)
    scale = eye_distance / current_eye_dist
    mean_color = np.mean(crop_img, axis=(0, 1)).astype(int)
    # center = (crop_img.shape[1]//2,crop_img.shape[0]//2)
    RotationMatrix = cv2.getRotationMatrix2D(eye_center,angel,scale)
    tX = output_size[0] / 2
    tY = output_size[1] / 3
    RotationMatrix[0, 2] += (tX - eye_center[0])
    RotationMatrix[1, 2] += (tY - eye_center[1])
    border_value = tuple(int(c) for c in mean_color)
    new_img = cv2.warpAffine(crop_img,RotationMatrix,output_size,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)
    return new_img


raw_dir = "人脸识别数据集/105_classes_pins_dataset/"
aligned_dir = "aligned_faces/"
os.makedirs(aligned_dir, exist_ok=True)
for img_name in os.listdir(raw_dir):
    imgs_path = os.path.join(raw_dir, img_name)
    for path in os.listdir(imgs_path): # path = 'Tom Holland245_4813.jpg'
        one_img_path = os.path.join(imgs_path,path)
        print(path)
        crop_img,probs,landmarks = face_detect(one_img_path)
        new_image = Alignment(crop_img,landmarks)
        if new_image is None:
            continue
        new_aligned_dir = os.path.join(aligned_dir,img_name)
        os.makedirs(new_aligned_dir, exist_ok=True)
        save_path = os.path.join(new_aligned_dir, path)
        # abs_save_path = os.path.abspath(save_path)
        # print(f"保存路径: {save_path}")
        # print(f"保存路径（绝对路径）: {abs_save_path}")
        bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_image)
# ax[1].imshow(crop_image)
# plt.show()
#         plt.imshow(new_image)
#         plt.show()