import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib

from vgg19_trainable import Vgg19 as vgg19_train
from vgg19_forcastable import Vgg19 as vgg19_forcast

BATCH_SIZE = 10
Class_Nums = 2

# 该函数用于统计 TFRecord 文件中的样本数量(总数)
def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums


# 生成tfrecords文件
def gen_dogVScat_VGG19_tfrecords(path):
    tf_writer = tf.python_io.TFRecordWriter(path)
    for file in os.listdir("/home/taoming/data/dogAndCat2/train/"):
        if not file.endswith('jpg'):
            continue

        if file.startswith("dog"):
            label = 1
        else:
            label = 0

        file_path = "/home/taoming/data/dogAndCat2/train/" + file
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()

        example = tf.train.Example()
        feature = example.features.feature
        feature['img_data'].bytes_list.value.append(img_raw)
        feature['label'].int64_list.value.append(label)
        tf_writer.write(example.SerializeToString())

    tf_writer.close()

def read_and_decode_dogVScat_VGG19(filename_queue):
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
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return img, label


def train_data(image_record_path):
    batch_num = total_sample(image_record_path) / BATCH_SIZE
    print("batch_num:{}".format(batch_num))

    filename_queue = tf.train.string_input_producer([image_record_path], shuffle=True)

    image, label = read_and_decode_dogVScat_VGG19(filename_queue)

    image_train, label_train = tf.train.shuffle_batch([image, label], BATCH_SIZE, num_threads=1,
                                                      capacity=5 + BATCH_SIZE * 3, min_after_dequeue=5)

    train_labels_one_hot = tf.one_hot(label_train, 2, on_value=1.0, off_value=0.0)
    x_data = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y_target = tf.placeholder(tf.float32, shape=[None, Class_Nums])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=True,
                                  dtype=tf.int32)

    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19_train('/home/taoming/data/vgg19.npy')
    vgg.build(x_data, train_mode)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=vgg.prob))
    optimizer = tf.train.AdamOptimizer(10e-5).minimize(loss, global_step=global_step)

    train_correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(y_target, 1))

    train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    loss_list = []
    acc_list = []

    try:
        for i in range(100):
            cost_avg = 0
            acc_avg = 0
            #print("xxxx:" + str(i))
            for j in range(int(batch_num)):
                #print(1)
                image_batch, label_batch = session.run([image_train, train_labels_one_hot])
                _, step, acc, cost = session.run([optimizer, global_step, train_accuracy, loss],
                                                 feed_dict={x_data: image_batch, y_target: label_batch,train_mode:True})
                acc_avg += (acc / batch_num)
                cost_avg += (cost / batch_num)
                #print("acc_avg:{}".format(acc_avg))
                #print("cost_avg:{}".format(cost_avg))
            print("step %d, training accuracy %0.10f loss %0.10f" % (i, acc_avg, cost_avg))
            loss_list.append(cost_avg)
            acc_list.append(acc_avg)
            #saver.save(session, 'model.ckpt', global_step=i)
            vgg.save_npy(session, './final.npy')
    except tf.errors.OutOfRangeError:
        print('Done training --epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    session.close()


def load_image_forcast(image_path):
    image = Image.open(image_path)
    image = image.resize([224, 224])
    image = np.array(image)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [-1, 224, 224, 3])
    image = tf.cast(image,tf.float32)* (1. / 255)
    return image

def forcast_dirs_image(dir,model_path):
    label_list = []
    image_list = []
    right = 0
    error = 0
    for file in os.listdir(dir):
        if not file.endswith('jpg'):
            continue
        if file.startswith("dog"):
            label = 1
        else:
            label = 0
        label_list.append(label)
        file_path = dir + file
        image = load_image_forcast(file_path)
        image_list.append(image)
    n = len(image_list)
    image_batch = tf.concat(image_list,axis=0)
    vgg = vgg19_forcast.Vgg19(model_path)
    vgg.build(image_batch)
    with tf.Session() as sess:
        result = sess.run(vgg.prob)
        for i in range(len(label_list)):
            result_i = np.argmax(result[i])
            if result_i == label_list[i]:
                right += 1
            else:
                error += 1
    print("right:{}".format(right))
    print("error:{}".format(error))



if __name__ == "__main__":
    #gen_dogVScat_VGG19_tfrecords('cat_vs_dog_vgg19.tfrecord')
    train_data("cat_vs_dog_vgg19.tfrecord")
    #forcast_dirs_image('/home/taoming/data/dogAndCat2/test2/','./final.npy')


