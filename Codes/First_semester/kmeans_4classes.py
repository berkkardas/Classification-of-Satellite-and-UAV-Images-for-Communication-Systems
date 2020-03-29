from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import h5py
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
file_path = os.path.dirname(os.path.abspath(__file__))
train = file_path + "/900Mhz_300m_imWithHisto_train_data.h5"
test = file_path + "/900Mhz_300m_imWithHisto_test_data.h5"

with h5py.File(train, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]
    # Get the data
    images = list(f[a_group_key])
    labels = list(f[b_group_key])

with h5py.File(test, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]
    # Get the data
    images.extend(list(f[a_group_key]))
    labels.extend(list(f[b_group_key]))


#from sklearn.preprocessing import normalize
#data_scaled = normalize(data)

#data_scaled = data
#data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
# data_scaled.head()

#import scipy.cluster.hierarchy as shc
#plt.figure(figsize=(10, 7))
# plt.title("Dendrograms")
#dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
#

k = 4

cluster = AgglomerativeClustering(
n_clusters=k, affinity='euclidean', linkage='ward')
cluster.fit_predict(labels)
hia_labels = cluster.labels_

#kmeans = KMeans(n_clusters=k).fit(labels)
#print(kmeans.labels_)
#kmeans_labels = kmeans.labels_

# print(silhouette_score(labels, hia_labels))
# print(silhouette_score(labels, kmeans_labels))


# for i in range(999):
#    img = array_to_img(images[i])
#    directory = file_path + '/hia_4class/' + str(hia_labels[i]) + "/"
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    img.save(directory + str(i) + '.png')
#

print("Keys : {}".format(labels))

for i in range(999):
   img = array_to_img(images[i])
   directory = file_path + '/hia_4class2/' + str(hia_labels[i]) + "/"
   if not os.path.exists(directory):
       os.makedirs(directory)
   img.save(directory + str(i) + '.png')

   a = labels[i]

   x = [0,1,2,3,4,5,6,7]
   y = list(a)

   plt.title("Histogram")
   plt.xlabel("Bins")
   plt.ylabel("dB")
   plt.bar(x,y)

   plt.savefig(directory + str(i) + 'h' + '.png')
   plt.clf()




   
