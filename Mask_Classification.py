import os
import numpy as np 
import pandas as pd 
import xml.etree.ElementTree as et
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import random as rand

"""# Import Data

# Data Processing 
Extract faces from images
"""

image_path = "/kaggle/input/face-mask-detection/images"
label_path = "/kaggle/input/face-mask-detection/annotations"

dic = {"image": [],"Dimensions": []}
for i in range(1,116):
	dic[f'Object {i}']=[]
print("Generating data in CSV format....")

for file in os.listdir(label_path):
    row = []
    xml = et.parse(label_path+'/'+file) 
    root = xml.getroot()
    img = root[1].text
    row.append(img)
    h,w = root[2][0].text,root[2][1].text
    row.append([h,w])

    for i in range(4,len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        row.append(temp)
    for i in range(len(row),119):
        row.append(0)
    for i,each in enumerate(dic):
      dic[each].append(row[i])
df = pd.DataFrame(dic)
df.head()

image_directories = sorted(glob.glob(os.path.join(image_path,"*.png")))

j=0
classes = ["without_mask","mask_weared_incorrect","with_mask"]
labels = []
data = []

for idx,image in enumerate(image_directories):
    img  = cv2.imread(image)
    #scale to dimension
    X,Y = df["Dimensions"][idx]
    cv2.resize(img,(int(X),int(Y)))
    #find the face in each object
    for obj in df.columns[2:]:
        info = df[obj][idx]
        #print(obj)
        #print(idx)
        if info!=0:
            label = info[0]
            info[0] = info[0].replace(str(label), str(classes.index(label)))
            info=[int(each) for each in info]
            face = img[info[2]:info[4],info[1]:info[3]]
            if((info[3]-info[1])>20 and (info[4]-info[2])>20):
                try:
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    data.append(face)
                    labels.append(label)
                    if(label=="mask_weared_incorrect"):
                        data.append(face)
                        labels.append(label)

                except:
                    pass

data = np.array(data, dtype="float32")
labels = np.array(labels)
labels

print(len(labels))
print(np.unique(labels,return_counts=True))

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

""" ResNet50 """

lr = 1e-4
epochs = 50
batch = 1

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.3, stratify=labels, random_state=42)

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
# from google.colab.patches import cv2_imshow
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from tensorflow.keras.applications import ResNet50

pretrainedResNet50 = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))


pretrainedResNet50Model = pretrainedResNet50.output
pretrainedResNet50Model = AveragePooling2D(pool_size=(7, 7))(pretrainedResNet50Model)
pretrainedResNet50Model = Flatten(name="flatten")(pretrainedResNet50Model)
pretrainedResNet50Model = Dense(64, activation="relu")(pretrainedResNet50Model)
pretrainedResNet50Model = Dropout(0.5)(pretrainedResNet50Model)
pretrainedResNet50Model = Dense(3, activation="softmax")(pretrainedResNet50Model)

resNet50 = Model(inputs=pretrainedResNet50.input, outputs=pretrainedResNet50Model)

for layer in pretrainedResNet50.layers:
	layer.trainable = False

opt = Adam(lr)
resNet50.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
resnet50pretrained_history = resNet50.fit(x=trainX, y=trainY, batch_size=batch, steps_per_epoch=len(trainX) // batch,
	validation_data=(testX, testY),validation_steps=len(testX) // batch, epochs=epochs)

result_resnet50pretrained=pd.DataFrame(resnet50pretrained_history.history)
result_resnet50pretrained

predIdxs_resnet = resNet50.predict(testX, batch_size=32)

predIdxs_resnet = np.argmax(predIdxs_resnet, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs_resnet,
	target_names=lb.classes_))

resNet50.evaluate(testX, testY)

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), resnet50pretrained_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), resnet50pretrained_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), resnet50pretrained_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), resnet50pretrained_history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

""" VGG16 """

from tensorflow.keras.applications import VGG16
pretrainedVGG16 = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))


pretrainedVGG16Model = pretrainedVGG16.output
pretrainedVGG16Model = AveragePooling2D(pool_size=(7, 7))(pretrainedVGG16Model)
pretrainedVGG16Model = Flatten(name="flatten")(pretrainedVGG16Model)
pretrainedVGG16Model = Dense(64, activation="relu")(pretrainedVGG16Model)
pretrainedVGG16Model = Dropout(0.5)(pretrainedVGG16Model)
pretrainedVGG16Model = Dense(3, activation="softmax")(pretrainedVGG16Model)

vgg16 = Model(inputs=pretrainedVGG16.input, outputs=pretrainedVGG16Model)

for layer in pretrainedVGG16.layers:
	layer.trainable = False

vgg16.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
vgg16pretrained_history = vgg16.fit(x=trainX, y=trainY, batch_size=batch, steps_per_epoch=len(trainX) // batch,
	validation_data=(testX, testY),validation_steps=len(testX) // batch, epochs=epochs)

result_vgg16pretrained=pd.DataFrame(vgg16pretrained_history.history)
result_vgg16pretrained

predIdxs_VGG = vgg16.predict(testX, batch_size=32)

predIdxs_VGG = np.argmax(predIdxs_VGG, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs_VGG,
	target_names=lb.classes_))

vgg16.evaluate(testX, testY)


""" MobileNetV2 """

MobileNetModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

MobileNetpretrainedModel = MobileNetModel.output
MobileNetpretrainedModel = AveragePooling2D(pool_size=(7, 7))(MobileNetpretrainedModel)
MobileNetpretrainedModel = Flatten(name="flatten")(MobileNetpretrainedModel)
MobileNetpretrainedModel = Dense(64, activation="relu")(MobileNetpretrainedModel)
MobileNetpretrainedModel = Dropout(0.5)(MobileNetpretrainedModel)
MobileNetpretrainedModel = Dense(3, activation="softmax")(MobileNetpretrainedModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
mobilenet = Model(inputs=MobileNetModel.input, outputs=MobileNetpretrainedModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in MobileNetModel.layers:
	layer.trainable = False

# aug = ImageDataGenerator(
#     zoom_range=0.1,
#     rotation_range=25,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest"
#     )

mobilenet.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
mobilenetpretrained_history = mobilenet.fit(x=trainX, y=trainY, batch_size=batch, steps_per_epoch=len(trainX) // batch,
	validation_data=(testX, testY),validation_steps=len(testX) // batch, epochs=epochs)

mobilenetpretrained=pd.DataFrame(mobilenetpretrained_history.history)
mobilenetpretrained

predIdxs_mobilenet = mobilenet.predict(testX, batch_size=32)

predIdxs_mobilenet = np.argmax(predIdxs_mobilenet, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs_mobilenet,
	target_names=lb.classes_))

mobilenet.evaluate(testX, testY)
