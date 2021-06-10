import codecs
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import tifffile as tiff
import numpy as np


def load_string_list(file_path, is_utf8=False):
    """
    Load string list from mitok file
    """
    try:
        if is_utf8:
            f = codecs.open(file_path, 'r', 'utf-8')
        else:
            f = open(file_path)
        l = []
        for item in f:
            item = item.strip()
            if len(item) == 0:
                continue
            l.append(item)
        f.close()
    except IOError:
        print('open error %s' % file_path)
        return None
    else:
        return l


def parse_each_line(each_line):

    pred_label = int(each_line.split('/')[-1])

    patch_name = each_line.split('/')[0].split('.')[0].split('_')

    index = int(patch_name[1])
    i = int(patch_name[2])
    j = int(patch_name[3])

    return index, i, j, pred_label


def fuseMask(img, color, alpha):
    '''
    color: (R, G, B)
    '''
    height, width, depth = img.shape

    mask = Image.new("RGB", (height, width), color)

    mask = np.array(mask)

    fused = cv2.addWeighted(img, alpha, mask, (1-alpha), 0)

    return fused

if __name__ == '__main__':
    tif_file_path = '*/*.tif'#要分类的子图
    img = tiff.imread(tif_file_path)[:, :, :3]
    # img.shape
    # 设置融合参数，依据需求确定不透明度
    red = (255, 0, 0)
    # blue = (0, 0, 255)
    alpha = 0.1
    #预测图上色
    for m in range(1, 4):
        for n in range(1,4):
            result_img = np.zeros(img.shape, dtype='uint8')
            patch_save_path='*/*pr'+str(m)+'c'+str(n)+'/'#预保存文件名的prefix
            result_txt_path = '*/*.txt'#对应子图的patch分类结果
            result_txt = load_string_list(result_txt_path)
            # print('正在输出:'+'r'+str(m)+'c'+str(n)+'_predict_non.png')
            for k in tqdm(range(len(result_txt)),postion=0):
                each_line = result_txt[k]
                index, i, j, pred_label = parse_each_line(each_line)
                #print(index,i,j,pred_label)
                img_patch_name = 'patch' + '_' + str(index) + '_' + str(i) + '_' + str(j) + '.png'
                #print(img_patch_name)
                img_patch_path = patch_save_path + '/' + img_patch_name
                #print(img_patch_path)
                tmp_img_patch = cv2.imread(img_patch_path)
                tmp_img_patch = cv2.cvtColor(tmp_img_patch, cv2.COLOR_BGR2RGB)
                
                if pred_label == 0:  # non-UV
                    tmp_result = tmp_img_patch
                if pred_label == 1:  # UrbanVillage      
                    tmp_result = fuseMask(tmp_img_patch, red, alpha)

                
                result_img[i*224:(i+1)*224, j*224:(j+1)*224,] = tmp_result
            result_img_RGB = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('*/*.png', result_img_RGB)#保存分类结果图
