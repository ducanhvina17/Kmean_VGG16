import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Flatten, Input
import numpy as np 
from sklearn.cluster import KMeans
import pickle
import numpy as np
import os
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
start_time = time.time()
vgg16_feature_list=[]
NUM_CLUSTERS=256
trainfile=open('database/train.txt','r')
listfiletrain=(trainfile.read().split())

for img_path in listfiletrain:
    if (img_path[img_path.rfind('.')+1:]=="jpg"):

        dirfile=("features/vgg16_fc2/"+(img_path[img_path.find('/')+1:img_path.rfind('/')]))
        filename=((img_path[img_path.rfind('/')+1:]))
        img_path=dirfile+"/"+filename
        vgg16_feature_list.append(np.load(img_path+".npy").flatten())

vgg16_feature_list_np = np.array(vgg16_feature_list)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(vgg16_feature_list_np) 
# save the model to disk
filename = 'finalized_modelKmean.sav'
pickle.dump(kmeans, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# print(loaded_model.labels_)
# print(loaded_model.cluster_centers_)

print("--- %s seconds ---" % (time.time() - start_time))