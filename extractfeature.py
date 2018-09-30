import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Flatten, Input
import numpy as np
import os
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
start_time = time.time()
model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=model.input,outputs=model.get_layer('fc2').output)

model.summary()
trainfile=open('database/train.txt','r')
listfiletrain=(trainfile.read().split())

for img_path in listfiletrain:
    if (img_path[img_path.rfind('.')+1:]=="jpg"):

        dirfile=("features/vgg16_fc2/"+(img_path[img_path.find('/')+1:img_path.rfind('/')]))
        filename=((img_path[img_path.rfind('/')+1:]))
        if not os.path.exists(dirfile):
            os.makedirs(dirfile)
            print(dirfile)   
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = model.predict(img_data)
        np.save(dirfile+"/"+filename, vgg16_feature)
print("--- %s seconds ---" % (time.time() - start_time))