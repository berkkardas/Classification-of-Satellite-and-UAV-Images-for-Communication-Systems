import h5py
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
file_path = os.path.dirname(os.path.abspath(__file__))


pic = plt.imread(file_path + "/input/266.png") 
print(pic.shape)
plt.imshow(pic)
pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)


#from sklearn.preprocessing import normalize
#data_scaled = normalize(data)

#data_scaled = data
#data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
#data_scaled.head()

#import scipy.cluster.hierarchy as shc
#plt.figure(figsize=(10, 7))  
#plt.title("Dendrograms")  
#dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
#
    
    
#from sklearn.cluster import AgglomerativeClustering
#cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
#cluster.fit_predict(labels2)
#print(cluster.labels_)
#hia_labels = cluster.labels_
#
#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=4).fit(labels2)
#print(kmeans.labels_)
#kmeans_labels = kmeans.labels_
#
#from sklearn.metrics import silhouette_score
#print(silhouette_score(labels2, hia_labels))
#print(silhouette_score(labels2, kmeans_labels))

#
#for i in range(999):
#    img = array_to_img(images[i])
#    directory = file_path + '/hia_4class/' + str(hia_labels[i]) + "/"
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    img.save(directory + str(i) + '.png')
#    
#for i in range(999):
#    img = array_to_img(images[i])
#    directory = file_path + '/kmeans_4class/' + str(kmeans_labels[i]) + "/"
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    img.save(directory + str(i) + '.png')



