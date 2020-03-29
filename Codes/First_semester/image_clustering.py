from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path

file_path = os.path.dirname(os.path.abspath(__file__))
image.LOAD_TRUNCATED_IMAGES = True 
model = ResNet50(weights='imagenet', include_top=False)

# Variables
imdir = file_path + "/input/"
targetdir = file_path + "/output/"
number_clusters = 4

# Loop over files and get features
filelist = glob.glob(os.path.join(imdir, '*.png'))
filelist.sort()
featurelist = []
for i, imagepath in enumerate(filelist):
    print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters).fit(np.array(featurelist))

# Copy images renamed by cluster 
# Check if target dir exists
try:
    os.makedirs(targetdir)
except OSError:
    pass
# Copy with cluster name
print("\n")
for i, m in enumerate(kmeans.labels_):
    directory = targetdir + str(m) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
    shutil.copy(filelist[i], directory + str(i) + ".png")
    
    
#import matplotlib.pyplot as plt
#import scipy.cluster.hierarchy as shc
#plt.figure(figsize=(10, 7))  
#plt.title("Dendrograms")  
#dend = shc.dendrogram(shc.linkage(np.array(featurelist), method='ward'))
#
#from sklearn.cluster import AgglomerativeClustering
#cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
#cluster.fit_predict(np.array(featurelist))
#print(cluster.labels_)
#lll = cluster.labels_
#lll2 = kmeans.labels_

from sklearn.metrics import silhouette_score
print(silhouette_score(np.array(featurelist), kmeans.labels_))

## calculate distortion for a range of number of cluster
#distortions = []
#for i in range(1, 8):
#    km = KMeans(
#        n_clusters=i, init='random',
#        n_init=10, max_iter=300,
#        tol=1e-04, random_state=0
#    )
#    km.fit(np.array(featurelist))
#    distortions.append(km.inertia_)
#    print(i)
#
## plot
#plt.plot(range(1, 8), distortions, marker='o')
#plt.xlabel('Number of clusters')
#plt.ylabel('Distortion')
#plt.show()
    
    
    