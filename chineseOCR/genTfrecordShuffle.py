import os
import random
import shutil
import math
from PIL import Image
import tensorflow as tf


def read_file_to_list(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('png'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                #print (file_path)
            else:
                print(file)
    random.shuffle(file_list)
    return file_list



def according_file_path_return_label(file_path):
    print(file_path)
    m = file_path.split('/')
    label = int(m[6])
    return label



def generate_tfrecords(file_path_list, tf_record_image_number,tfrecord_dir):
    tfrecord_number = math.ceil(len(file_path_list) / tf_record_image_number)
    for i in range(tfrecord_number):
        if i == (tfrecord_number - 1):
            tf_record_image_list = file_path_list[i*tf_record_image_number:-1]
        else:
            tf_record_image_list = file_path_list[i*tf_record_image_number:(i + 1)* tf_record_image_number]

        tfrecord_path = tfrecord_dir + 'chineseOCR_vgg19_{}.tfrecord'.format(i)
        tf_writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for file in tf_record_image_list:
            label = according_file_path_return_label(file)
            img = Image.open(file)
            img = img.resize((224, 224))
            img_raw = img.tobytes()

            example = tf.train.Example()
            feature = example.features.feature
            feature['img_data'].bytes_list.value.append(img_raw)
            feature['label'].int64_list.value.append(label)
            tf_writer.write(example.SerializeToString())

        tf_writer.close()





if __name__ == "__main__":
    file_list = read_file_to_list("/home/taoming/data/chineseOCR/testSample")
    generate_tfrecords(file_list,10000,'/home/taoming/data/chineseOCR/testSampleTfrecords/')
