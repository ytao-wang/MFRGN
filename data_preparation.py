import imageio
import os
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sample4geo.transforms import Cut


############################ Prepare Images in VIGOR to Accelerate Training Time #############################
name = 'Chicago, NewYork, SanFrancisco, Seattle'
city = 'SanFrancisco'
mode = 'ground'  # panorama or satellite 
# input_dir = f'/mnt/wangyuntao/Datasets/VIGOR/{city}/satellite/'
# output_dir = f'/mnt/wangyuntao/Datasets/VIGOR_processed/satellite/{city}/'
input_dir = f'/mnt/wangyuntao/Datasets/VIGOR/{city}/panorama/'
output_dir = f'/mnt/wangyuntao/Datasets/VIGOR_processed_320/ground/{city}/'

image_size_sat = (320, 320)
img_size_ground = (320, 640)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

grd_transforms = A.Compose([Cut(cutting=0, p=1.0),
                            A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_AREA, p=1.0),
                            A.Normalize(mean, std),
                            ToTensorV2(),
                            ])

sat_transforms = A.Compose([
                            A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_AREA, p=1.0),
                            # A.Normalize(mean, std),
                            # ToTensorV2(),
                            ])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

print("Resize and process images to .pt for VIGOR")
for img in tqdm(images):
    img = cv2.imread(input_dir + img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = grd_transforms(image=img)['image'] # when img is sat, use img = sat_transforms(image=img)['image']

    path = output_dir + img

    path = path.replace('.jpg', '.pt') if '.jpg' in path else path.replace('.png', '.pt')
    with open(path, 'wb') as _f1:
            pickle.dump(img, _f1)



# ############################ Prepare Images in CVACT to Accelerate Training Time (no crop) #############################
input_dir = '/mnt/wangyuntao/Datasets/CVACT/ANU_data_small/satview_polish/'
output_dir = '/mnt/wangyuntao/Datasets/CVACT_processed/CVACT/ANU_data_smal/satview_polish/'

image_size_grd = (128, 512)
image_size_sat = (256, 256)
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

grd_transforms = A.Compose([Cut(cutting=0, p=1.0),
                            A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_AREA, p=1.0),
                            # A.Normalize(mean, std),
                            # ToTensorV2(),
                            ])

sat_transforms = A.Compose([
                            A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_AREA, p=1.0),
                            # A.Normalize(mean, std),
                            # ToTensorV2(),
                            ])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
images = os.listdir(input_dir)

print("Resize and process images to .pt for CVACT")
for img in tqdm(images):
    img1 = cv2.imread(input_dir + img)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = img1

    img1 = sat_transforms(image=img1)['image']  # when img is grd, use img = grd_transforms(image=img)['image']

    path = output_dir + img
    path = path.replace('.jpg', '.pt') if '.jpg' in path else path.replace('.png', '.pt')
    with open(path, 'wb') as _f1:
            pickle.dump(img1, _f1)


############################ Prepare Images in CVACT to Accelerate Training Time (with crop) #############################
input_dir = '/mnt/wangyuntao/Datasets/CVACT/ANU_data_small/streetview/'
output_dir = '/mnt/wangyuntao/Datasets/CVACT/ANU_data_small/streetview_processed/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = os.listdir(input_dir)

print("Resize and process images for CVACT")
for img in tqdm(images):
    signal = imageio.imread(input_dir + img)

    start = int(832 / 4)
    image = signal[start: start + int(832 / 2), :, :]
    image = cv2.resize(image, (512, 128), interpolation=cv2.INTER_AREA)
    imageio.imwrite(output_dir + img.replace('.jpg', '.png'), image)

    
