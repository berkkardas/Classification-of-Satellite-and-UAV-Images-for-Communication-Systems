from PIL import Image
import os
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.abspath(__file__))
in_path = file_path + '/in'
out_path = file_path + '/out'

for entry in os.scandir(in_path):
    if entry.path.endswith(".jpg") and entry.is_file():
        print(entry.name.lstrip('0'))
        img = Image.open(entry.path)
        img.save(out_path + '/' + entry.name.lstrip('0'))
