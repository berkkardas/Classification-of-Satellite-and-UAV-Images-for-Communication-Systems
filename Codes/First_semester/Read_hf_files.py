import h5py
import os
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
file_path = os.path.dirname(os.path.abspath(__file__))
filename = file_path + "/900Mhz_300m_imWithHisto_train_data.h5"

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]

    # Get the data
    images = list(f[a_group_key])
    labels = list(f[b_group_key])


data = pd.DataFrame(labels)

#from sklearn.preprocessing import normalize
#data_scaled = normalize(data)
data_scaled = data
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)
print(cluster.labels_)
lll = cluster.labels_

for i in range(700):
    img = array_to_img(images[i])
    directory = file_path + '/train/' + str(lll[i]) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.save(directory + str(i) + '.png')



