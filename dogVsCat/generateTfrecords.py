import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 生成tfrecords文件
def gen_dogVScat_tfrecords(path):
    tf_writer = tf.python_io.TFRecordWriter(path)
    for file in os.listdir("/Users/taoming/data/dogVScat/kaggle/train/"):
        if not file.endswith('jpg'):
            continue

        if file.startswith("dog"):
            label = 1
        else:
            label = 0

        file_path = "/Users/taoming/data/dogVScat/kaggle/train/" + file
        img = Image.open(file_path)
        img = img.resize((180, 180))
        img_raw = img.tobytes()

        example = tf.train.Example()
        feature = example.features.feature
        feature['img_data'].bytes_list.value.append(img_raw)
        feature['label'].int64_list.value.append(label)
        tf_writer.write(example.SerializeToString())

    tf_writer.close()


# 读取tfrecords文件
def read_and_decode_dogVScat(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    img = tf.decode_raw(features['img_data'], tf.uint8)
    img = tf.reshape(img, [180, 180, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #此处由于是图片还原，所以不需要归一化处理
    #img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return img, label

def read_and_decode_dogVScat_change(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_data': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    img = tf.decode_raw(features['img_data'], tf.uint8)
    img = tf.reshape(img, [180, 180, 3])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #此处由于是图片还原，所以不需要归一化处理
    #也不应该有下句，处理成tensor张量就无法处理了
    #img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return img, label

#该函数用于统计 TFRecord 文件中的样本数量(总数)
def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums


# 读取数据还原数据
def restore_image_from_tfrecords(tfrecord_path):
    total_sample_num = total_sample(tfrecord_path)
    filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle=True)
    image, label = read_and_decode_dogVScat_change(filename_queue)
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            image_run, label_run = sess.run([image, label])
            img = Image.fromarray(image_run,"RGB")
            img.save("/Users/taoming/data/dogVScat/kaggle/restore/" + str(i) + '_''Label_' + str(label_run) + '.jpg')
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    #gen_dogVScat_tfrecords('cat_vs_dog.tfrecord')
    restore_image_from_tfrecords('cat_vs_dog.tfrecord')