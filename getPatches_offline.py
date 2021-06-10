
import numpy as np
import tifffile as tiff
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_img_patch(img_patch, save_path, index, i, j):
    img_patch_name = 'patch' + '_' + str(index) + '_' + str(i) + '_' + str(j) + '.png'
    
    img_patch_save_path = save_path + '/' + img_patch_name
    cv2.imwrite(img_patch_save_path, img_patch)
    return True

if __name__ == '__main__':
    save_path = '*/L17/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #载入影像
    tif_file_path = '*/level17.tif'
    height, width, depth = img.shape
    img = tiff.imread(tif_file_path)
    #计算切片数量
    i_total = height//224#87
    j_total = width//224#88

    tmp_index = -1
    for i in tqdm(range(i_total)):
        #print(i)
        for j in range(j_total):
            #print(i,j)
            tmp_img_patch = img[i*224:(i+1)*224, j*224:(j+1)*224,]
            tmp_img_patch_rgb = cv2.cvtColor(tmp_img_patch, cv2.COLOR_BGR2RGB)
            
            tmp_index += 1
            save_img_patch(tmp_img_patch_rgb, save_path, tmp_index, i, j)