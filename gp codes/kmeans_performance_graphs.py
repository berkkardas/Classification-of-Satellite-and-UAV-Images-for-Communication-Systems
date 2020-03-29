import h5py
import os
from PIL import Image
import numpy as np
# from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# from keras.preprocessing.image import array_to_img
file_path = os.path.dirname(os.path.abspath(__file__))
filename = file_path + "/900Mhz_300m_imWithHisto_train_data.h5"
from sklearn.metrics import silhouette_score

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]

    # Get the data
    images = list(f[a_group_key])
    labels = list(f[b_group_key])





data = pd.DataFrame(labels)
# from sklearn.preprocessing import normalize
# data_scaled = normalize(data)
# data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
# data_scaled.head()

# df = data_scaled
df = data

K = range(2,12,1)
distortions = []
silhouette = []
db_score = []
calinski = []
for k in K:
    print("For {} clusters".format(k))
    km = KMeans(n_clusters=k, random_state=0).fit(df)
    labels = km.labels_
    # labels = km.predict(df)
    # print(labels[0])
    
    print("Silhouette Score: {}".format(silhouette_score(df, labels, metric='euclidean')))
    print("DB Score: {}".format(davies_bouldin_score(df, labels)))
    print("Calinski-Harabasz Index: {}".format(metrics.calinski_harabasz_score(df, labels)))
    distortions.append(km.inertia_)
    silhouette.append(silhouette_score(df, labels, metric='euclidean'))
    db_score.append(davies_bouldin_score(df, labels))
    calinski.append(metrics.calinski_harabasz_score(df, labels))


# plt.plot(K,silhouette,K,db_score)
# plt.plot(calinski)
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.legend(['Silhouette Score','DB Score'])
# plt.xticks(K)
# plt.show()
# print(lll2[2])
# print(filename)
# for i in range(700):
#     img = array_to_img(images[i])
#     directory = file_path + '/train/' + str(labels[i]) + "/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     img.save(directory + str(i) + '.png')

fig = plt.figure()

fig.suptitle('Cluster Validation Quality Measures for K-Means Clustering', fontweight="bold")
plt.subplots_adjust(hspace=0.5, wspace= 0.4)

plt.subplot(2, 2, 1)
plt.plot(K, silhouette, color='green')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Silhoutte Score')
plt.xticks(K)

plt.subplot(2, 2, 2)
plt.plot(K,db_score, color='black')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('DB Score')
plt.xticks(K)

plt.subplot(2, 2, 3)
plt.plot(K, calinski, color='red')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Calinski-Harabasz Index')
plt.xticks(K)

plt.subplot(2, 2, 4)
plt.title('Elbow Method')
plt.plot(K, distortions, 'bx-')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Distortion')
plt.xticks(K)
plt.show()
