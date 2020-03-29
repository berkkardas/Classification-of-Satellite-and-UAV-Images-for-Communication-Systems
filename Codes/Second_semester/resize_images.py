from PIL import Image
import os
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.abspath(__file__))
in_path = file_path + '/in'
out_path = file_path + '/out'

for entry in os.scandir(in_path):
    if entry.path.endswith(".bmp") and entry.is_file():
        print(entry.name)
        img = Image.open(entry.path)  # image extension *.png,*.jpg
        new_width = 600
        new_height = 600
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        # format may what u want ,*.png,*jpg,*.gif
        out_name = out_path + '/' + entry.name
        img.save(out_name[:-3] + 'jpg')
