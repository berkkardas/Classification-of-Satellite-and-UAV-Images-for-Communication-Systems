from PIL import Image
import os
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.abspath(__file__))
path1 = file_path + '/satellite_images_600_600'
path2 = file_path + '/heat_images_600_600'
out_path = file_path + '/out'


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


for entry in os.scandir(path1):
    if entry.path.endswith(".jpg") and entry.is_file():
        print(entry.name)
        im1 = Image.open(entry.path)
        im2 = Image.open(path2 + '/' + entry.name)
        get_concat_h(im1, im2).save(out_path + '/' + entry.name)
