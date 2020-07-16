
import numpy as np
import cv2
import os
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="3"
def get_intensity_value(value, min_val, max_val):
    if np.isnan(value):
        value = 0
    if np.isnan(max_val):
        max_val = 0.0001
    if np.isnan(min_val):
        min_val = 0
    nor_val=255 * ((value - min_val) / (max_val - min_val))
    if np.isnan(nor_val):
        nor_val=np.nan_to_num(nor_val)
    else:
        nor_val=int(nor_val)
    return nor_val


def pca(version):
    outersize=150
    trimsize=70
    innersize=outersize-2*trimsize
    model=load_model('bestmodel'+version)
    predict_folder="ahte_line" + version + "_" + str(innersize)
    test_folder = 'ahte_test'
    output_layer=2
    model = model.layers[output_layer]

    os.makedirs(predict_folder, exist_ok=True)
    os.makedirs(os.path.join(predict_folder, 'cv2_vis1'), exist_ok=True)
    os.makedirs(os.path.join(predict_folder, 'cv2_vis2'), exist_ok=True)

    for imgp in os.listdir(test_folder):
        print(imgp)
        page = cv2.imread('{}/{}'.format(test_folder, imgp), 0)
        rows,cols=page.shape
        x=rows//innersize
        y=cols//innersize
        prows=(x+1)*innersize+2*trimsize
        pcols=(y+1)*innersize+2*trimsize
        ppage=np.zeros([prows,pcols])
        ppage[trimsize:rows+trimsize,trimsize:cols+trimsize]=page[:,:]
        predicted_patch = model.predict(np.zeros((1, outersize, outersize, 1)))
        predicted_img = np.zeros((x+1,y+1, predicted_patch.shape[1]), np.float32)


        for i in range(0,x+1):
            for j in range(0,y+1):
                patch=ppage[i*innersize:i*innersize+outersize,j*innersize:j*innersize+outersize]
                patch = np.expand_dims(patch, axis=0)
                patch = np.expand_dims(patch, axis=3)
                predicted_patch = model.predict(patch)[0]

                predicted_img[i, j, :] = predicted_patch

        pca = PCA(n_components=predicted_img.shape[2])

        features = predicted_img.reshape(-1, predicted_img.shape[2])
        pca_t_features = pca.fit(features).transform(features)
        pca_t_features = pca_t_features[:, :3]

        rgb = [[get_intensity_value(pca_t_features[i, 0], pca_t_features[:, 0].min(), pca_t_features[:, 0].max()),
                get_intensity_value(pca_t_features[i, 1], pca_t_features[:, 1].min(), pca_t_features[:, 1].max()),
                get_intensity_value(pca_t_features[i, 2], pca_t_features[:, 2].min(), pca_t_features[:, 2].max())]
                for i in range(pca_t_features.shape[0])]

        rgb = np.asarray(rgb, dtype=np.uint8).reshape((*predicted_img.shape[:2], 3))
        rgb_rows,rgb_cols,_=rgb.shape
        result=np.zeros([rows,cols,3])
        for i in range(rgb_rows):
            for j in range(rgb_cols):
                pixel_value=rgb[i,j,:]
                result[i*innersize:i*innersize+innersize,j*innersize:j*innersize+innersize,:]=pixel_value

        # big_rgb=cv2.resize(rgb,(page.shape[1]-pad_h,page.shape[0]-pad_w))
        # org_rgb=np.zeros([page.shape[0],page.shape[1],3])
        # org_rgb[:-pad_w,:-pad_h]=big_rgb


        cv2.imwrite('{}/{}'.format(os.path.join(predict_folder, 'cv2_vis1'), imgp), rgb)
        cv2.imwrite('{}/{}'.format(os.path.join(predict_folder, 'cv2_vis2'), imgp), result)


pca('7')
