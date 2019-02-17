import tensorflow as tf
import os
from PIL import Image

def read_and_decode(file_path):
    filename_queue = tf.train.string_input_producer([file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    if (features['label'].value_index) == 1:
        label = tf.cast([0,1], tf.float32)
    else:
        label = tf.cast([1, 0], tf.float32)
    return img, label


# 生成tfrecords文件
def gen_dogVScat_tfrecords(path):
    tf_writer = tf.python_io.TFRecordWriter(path)
    for file in os.listdir("/Users/taoming/Data/dogVScat/kaggle/train/"):
        if not file.endswith('jpg'):
            continue
        label = None
        if file.startswith("dog"):
            label = 1
        else:
            label = 0

        file_path = "/Users/taoming/Data/dogVScat/kaggle/train/" + file
        img = Image.open(file_path)
        img = img.resize((180, 180))
        img_raw = img.tobytes()

        example = tf.train.Example()
        feature = example.features.feature
        feature['img_data'].bytes_list.value.append(img_raw)
        feature['label'].int64_list.value.append(label)
        tf_writer.write(example.SerializeToString())

    tf_writer.close()



# if __name__ == "__main__":
#     gen_dogVScat_tfrecords('dogVScat.tfrecord')


