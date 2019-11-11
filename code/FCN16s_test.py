#FCN_16s
import cv2, os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, StratifiedKFold
## Import usual libraries
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from sklearn.utils import shuffle
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model



import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax


def give_color_to_seg_img(seg,n_classes):
    '''
    seg : (input_width,input_height,3)
    '''
    
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)


def loadImage( path , width , height ):
    image = cv2.imread(path, 1)
    image = cv2.resize(image, ( width , height ))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def loadLabel( path , nClasses ,  width , height  ):
    label = np.zeros((height, width , nClasses ))
    image = cv2.imread(path, 1)
    image = cv2.resize(image, ( width,height ))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_red = image[:, : , 0] # [width,height,channel]
    image_green = image[:, : , 1] # [width,height,channel]
    image_blue = image[:, : , 2] # [width,height,channel]
    for channel in range(nClasses):
        if channel == 0:
            label[: , : , channel ] = ((image_red == 255) & (image_green == 255) & (image_blue == 255)).astype(int) #name="generalface" 
        elif channel == 1:
            label[: , : , channel ] = ((image_red == 0) & (image_green == 0) & (image_blue == 0)).astype(int) #name="background"
        else:
            continue
    return label



def Contigency(nClasses,labels, predicts):
    matrixs = []
    for channel in range(nClasses):
        TP = np.sum( (labels == channel) & (predicts == channel) )
        TN = np.sum( (labels != channel) & (predicts != channel) )
        FP = np.sum( (labels != channel) & (predicts == channel) )
        FN = np.sum( (labels == channel) & (predicts != channel) )
        
        matrix = []
        matrix.append(TP)
        matrix.append(TN)
        matrix.append(FP)
        matrix.append(FN)

        matrixs.append(matrix)

    return matrixs





def Validate(matrixs):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    IoUs = []
    channel = 0
    for matrix in matrixs:
        TP = matrix[0]
        TN = matrix[1]
        FP = matrix[2]
        FN = matrix[3]
        IoU = TP/float(TP + FP + FN)
        acc = (TP+TN)/float(TP + TN + FP + FN)
        prec = TP/float(TP + FP)
        rec = TP/float(TP+FN)
        F1 = (2*prec*rec)/float(prec + rec)
        print("____________________________________________________________________")
        print("class {:02.0f}: #TP={:6.0f}, #TN={:6.0f}, #FP={:6.0f}, #FN={:5.0f}".format(channel,TP,TN,FP,FN))
        print("acc={:4.3f}, prec={:4.3f}, rec={:4.3f}, F1={:4.3f}, IoU={:4.3f}".format(acc,prec,rec,F1,IoU))
        channel = channel + 1
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))


def CRFs(images, y_preds, n_classes):
    X_i = images.astype(np.uint8)
    Y_i = y_preds
    transpose_matrix = np.array(Y_i).transpose((2, 0, 1))
    unary = unary_from_softmax(transpose_matrix, 0.7)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(X_i.shape[0], X_i.shape[1], n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=X_i, compat=10)
    Q = d.inference(20)
    arr = np.argmax(Q, axis=0)
    arr = arr.ravel().reshape((X_i.shape[0], X_i.shape[1]))
    result = np.zeros((X_i.shape[0], X_i.shape[1], n_classes ))
    for i in range(n_classes):
        result[:,:,i] = (arr==i).astype(int)
    
    return result






# load json and create model
json_file = open('/content/drive/My Drive/app/F7/output16/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
global model 
model = model_from_json(loaded_model_json)


# this is key : save the graph after loading the model
global graph
graph = tf.get_default_graph()


# load weights into new model
model.load_weights("/content/drive/My Drive/app/F7/output16/model.h5")
print("Loaded model from disk")


##### Read images in folder
location_image = "/content/drive/My Drive/app/F6/dataset1/images_prepped_test/"
location_segmentation = "/content/drive/My Drive/app/F6/dataset1/annotations_prepped_test/"

image_direction = os.listdir(location_image)
image_direction.sort()
segmentation_direction = os.listdir(location_segmentation)
segmentation_direction.sort()



##### Load and augment dataset 
n_classes = 2
input_width = 224
input_height = 224


X = []
Y = []

for real_pic , seg_pic in zip(image_direction,segmentation_direction) :
    pre_X = loadImage(location_image + real_pic, input_width, input_height)
    pre_Y = loadLabel(location_segmentation + seg_pic, n_classes, input_width , input_height)
    X.append(pre_X)
    Y.append(pre_Y)


train_rate = 0.85
##### Separate data
X, Y = np.array(X) , np.array(Y)
X_val, y_val = shuffle(X,Y)



##### Validation
print ("=================================================")
print ("Validation before CRFs")
print ("-------------------------")
print ("X_test.shape = ", X_val.shape)
print ("y_test.shape = ", y_val.shape)

print ("-------------------------")
# Validate by contigency
y_pred = model.predict(X_val)
y_predi = np.argmax(y_pred, axis=3)
sh = give_color_to_seg_img(y_predi[0],n_classes)
plt.imshow(sh)
plt.show()
y_testi = np.argmax(y_val, axis=3)
matrixs = Contigency(n_classes, y_testi, y_predi)
Validate(matrixs)

print ("-------------------------")
print ("=================================================")
print ("Validation after CRFs")
print ("-------------------------")
results = []
for i in range(len(X_val)):
    images = X_val[i]
    labels = y_pred[i]
    result = CRFs(images,labels,n_classes)    
    results.append(result)

results = np.asarray(results)
resultsi = np.argmax(results, axis=3)
sh = give_color_to_seg_img(resultsi[0],n_classes)
plt.imshow(sh)
plt.show()
matrixs = Contigency(n_classes, y_testi, resultsi)
Validate(matrixs)

print ("-------------------------")
print ("=================================================")

