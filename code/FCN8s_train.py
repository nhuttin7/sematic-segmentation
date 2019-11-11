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
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

from sklearn.utils import shuffle
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

def FCN(nClasses, input_width, input_height):

    ##########Input
    img_input = Input(shape=(input_width, input_height, 3))
    

    ##########Encode    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last')(x)
    block1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last')(block1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last')(x)
    block2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last')(block2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last')(x)
    block3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last')(block3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_last')(x)
    block4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last')(block4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_last')(x)
    block5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last')(x)

    


    ##########Decode
    out = Conv2D(4096, (7, 7), activation='relu', padding='same', name="conv6", data_format='channels_last')(block5)
    out = Dropout(0.5)(out)
    block7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name="conv7", data_format='channels_last')(out)
    block7 = Dropout(0.5)(block7)
    
    
    #upsampling for block7
    block7_up = Conv2DTranspose(nClasses, kernel_size=(4,4),  strides=(4,4), use_bias=False, data_format='channels_last')(block7)
    
    #up-upsampling for block4
    block4_up = Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="block4_pool_up", data_format='channels_last')(block4)
    block4_up_up = Conv2DTranspose(nClasses, kernel_size=(2,2), strides=(2,2), use_bias=False, data_format='channels_last')(block4_up)
    
    #upsampling for block3
    block3_up = Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="block3_pool_up", data_format='channels_last')(block3)



    
    ##########Add all    
    out = Add(name="add")([ block4_up_up, block3_up, block7_up ])
    out = Conv2DTranspose(nClasses, kernel_size=(8,8),  strides=(8,8), use_bias=False, data_format='channels_last')(out)
    out = (Activation('softmax'))(out)
    


    ##########Create model
    model = Model(img_input, out)

    return model



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




import random
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




##### GPU setting
'''
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = "2" 
set_session(tf.Session(config=config))
'''


##### Read images in folder
location_image = "/content/drive/My Drive/app/F6/dataset1/images_prepped_train/"
location_segmentation = "/content/drive/My Drive/app/F6/dataset1/annotations_prepped_train/"

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
X, Y = shuffle(X,Y)

index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
index_test  = list(set(range(X.shape[0])) - set(index_train))
                            
X, Y = shuffle(X,Y)
X_train, y_train = X[index_train],Y[index_train]
X_test, y_test = X[index_test],Y[index_test]

# Load fully convolution networks
model = FCN(nClasses = n_classes, input_width  = input_width, input_height = input_height)


# Optimize
sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())


#Fit the model
model.fit(X_train,y_train, validation_data=(X_test,y_test), batch_size=32, epochs=200, verbose=2)



##### Validation
print ("=================================================")
print ("Validation before CRFs")
print ("-------------------------")
print ("X_test.shape = ", X_test.shape)
print ("y_test.shape = ", y_test.shape)

print ("-------------------------")
# Accuracy of model
result = model.evaluate(X_test, y_test, verbose=2)
print("Test_Acc: %.2f%%" % (result[1]*100))

print ("-------------------------")
# MeanIU
y_pred = model.predict(X_test)
y_predi = np.argmax(y_pred, axis=3)
sh = give_color_to_seg_img(y_predi[0],n_classes)
plt.imshow(sh)
plt.show()
y_testi = np.argmax(y_test, axis=3)
#print (y_predi)
#sprint (y_pred.ravel().shape," ",y_predi.ravel().shape)
matrixs = Contigency(n_classes, y_testi, y_predi)
Validate(matrixs)

print ("-------------------------")
print ("=================================================")
print ("Validation after CRFs")
print ("-------------------------")
results = []
for i in range(len(X_test)):
    images = X_test[i]
    labels = y_pred[i]
    result = CRFs(images,labels,n_classes)    
    results.append(result)

results = np.asarray(results)
resultsi = np.argmax(results, axis=3)
sh = give_color_to_seg_img(resultsi[0],n_classes)
plt.imshow(sh)
plt.show()
#print (resultsi)
#print (results.ravel().shape," ",resultsi.ravel().shape)
matrixs = Contigency(n_classes, y_testi, resultsi)
Validate(matrixs)

print ("-------------------------")
print ("=================================================")




# To json file
cv_to_json = model.to_json()
with open("/content/drive/My Drive/app/F6/output/model.json", "w") as file:
    file.write(cv_to_json)
# To HDF5 file
model.save_weights("/content/drive/My Drive/app/F6/output/model.h5")
print("Saved model to disk")
