
from backbone import *
import os 

def predict(patchPath,model,threshold = 0.9):
    png_list = os.listdir()
    for i in tqdm(range(len(png_list)),position=0):

        each_png_path = pnglist[i]
        each_png = cv2.imread(each_png_path)
        each_png = cv2.cvtColor(each_png, cv2.COLOR_BGR2RGB)
        each_png = each_png.astype('float32')
        each_png = each_png/255.
        each_png = np.expand_dims(each_png, axis=0)
        
        Y_pred = model_dncnn.predict(each_png, verbose=1)
        if Y_pred[0,1]>=0.9:
            Y_predict=1
            each_result = png_list[i] + '/' + str(Y_predict)
        else:
            Y_predict=0
            each_result = png_list[i] + '/' + str(Y_predict)
        result_list.append(each_result)
    save_string_list('*/', result_list)#保存每张子图的分类结果

if __name__ == "__main__":
    path = '*/'#存放patcher的文件夹
    _weights = '*/UV_weights_new_Nadam100.h5'#94.27%,0.8839
    mmodel = DNCNN_non_local()
    mmodel.load_weights(_weights)
    predict(path,mmodel)
