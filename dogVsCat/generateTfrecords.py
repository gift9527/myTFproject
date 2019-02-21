import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 生成tfrecords文件
def gen_dogVScat_tfrecords(path):
    tf_writer = tf.python_io.TFRecordWriter(path)
    for file in os.listdir("/home/taoming/data/dogAndCat2/train/"):
        if not file.endswith('jpg'):
            continue
        label = None
        if file.startswith("dog"):
            label = 1
        else:
            label = 0

        file_path = "/home/taoming/data/dogAndCat2/train/" + file
        img = Image.open(file_path)
        img = img.resize((180, 180))
        img_raw = img.tobytes()

        example = tf.train.Example()
        feature = example.features.feature
        feature['img_data'].bytes_list.value.append(img_raw)
        feature['label'].int64_list.value.append(label)
        tf_writer.write(example.SerializeToString())

    tf_writer.close()

if __name__ == "__main__":
    gen_dogVScat_tfrecords('cat_vs_dog.tfrecord')
