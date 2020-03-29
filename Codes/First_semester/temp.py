import h5py
import os
from PIL import Image
import numpy as np
# from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
file_path = os.path.dirname(os.path.abspath(__file__))
filename = file_path + "/900Mhz_300m_imWithHisto_train_data.h5"
#filename = file_path + "/900Mhz_300m_imWithHisto_test_data.h5"

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]

    # Get the data
    images = list(f[a_group_key])
    labels = list(f[b_group_key])

#print(images[4].dtype)
#img = Image.fromarray(images[400].astype('uint16'), 'RGB')
# for i in range(300):
#     img = array_to_img(images[i])
#     img.save(file_path + '/images/test' + str(i) + '.png')

#w, h = 224, 224
#d = np.zeros((h, w, 3), dtype=np.uint8)
#img = Image.fromarray(d, 'RGB')


#print(im.shape)
#img = array_to_img(images[158])

# img.show()

#plt.imshow(img)
#plt.show()