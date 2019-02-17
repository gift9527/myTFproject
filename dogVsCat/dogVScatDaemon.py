import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Class_Nums = 2  # 共有 2 个种类
Sample_Nums = 10  # 每类取 10 个样本,共 50 个样本


# def load_images():
#     img_list = []
#     for i in range(Class_Nums):
#         path = 'F:\Python WorkSpace\FaceRecognize\\train\%d\\' % (i + 1)
#         for j in range(Sample_Nums):
#             file_name = '%03d.jpg' % (j + 1)
#             file = path + file_name
#             image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#             img_list.append(image)
#
#     return img_list


def get_accuracy(logits, targets):
    batch_prediction = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_prediction, targets))
    return 100. * num_correct / batch_prediction.shape[0]


# 以下代码用于实现卷积网络

def weight_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))


def bias_init(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def conv2d(input_data, conv_w):
    return tf.nn.conv2d(input_data, conv_w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input_data, size):
    return tf.nn.max_pool(input_data, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def conv_net(input_data):
    with tf.name_scope('conv1'):
        w_conv1 = weight_init([3, 3, 3, 8], 'conv1_w')  # 卷积核大小是 3*3 输入是 1 通道,输出为 8 通道,即提取8特征
        b_conv1 = bias_init([8], 'conv1_b')
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(input_data, w_conv1), b_conv1))
        bn1 = tf.contrib.layers.batch_norm(h_conv1)
        h_pool1 = max_pool(bn1, 2)

    with tf.name_scope('conv2'):
        w_conv2 = weight_init([5, 5, 8, 8], 'conv2_w')  # 卷积核大小是 5*5 输入是64,输出为 32
        b_conv2 = bias_init([8], 'conv2_b')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        bn2 = tf.contrib.layers.batch_norm(h_conv2)
        h_pool2 = max_pool(bn2, 2)

    with tf.name_scope('conv3'):
        w_conv3 = weight_init([5, 5, 8, 8], 'conv3_w')  # 卷积核大小是 5*5 输入是8,输出为 8
        b_conv3 = bias_init([8], 'conv3_b')
        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
        bn3 = tf.contrib.layers.batch_norm(h_conv3)
        h_pool3 = max_pool(bn3, 2)

    with tf.name_scope('fc1'):
        w_fc1 = weight_init([23 * 23 * 8, 120], 'fc1_w')  # 三层卷积后得到的图像大小为 22 * 22,共 50 个样本
        b_fc1 = bias_init([120], 'fc1_b')
        h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool3, [-1, 23 * 23 * 8]), w_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        w_fc2 = weight_init([120, Class_Nums], 'fc2_w')  # 将 130 个特征映射到 26 个类别上
        b_fc2 = bias_init([Class_Nums], 'fc2_b')
        h_fc2 = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

    return h_fc2


# 生成tfrecords文件
def gen_dogVScat_tfrecords(path):
    tf_writer = tf.python_io.TFRecordWriter(path)
    for file in os.listdir("/Users/closer/PycharmProjects/data/dogAndCat2/train/"):
        if not file.endswith('jpg'):
            continue
        label = None
        if file.startswith("dog"):
            label = 1
        else:
            label = 0

        file_path = "/Users/closer/PycharmProjects/data/dogAndCat2/train/" + file
        img = Image.open(file_path)
        img = img.resize((180, 180))
        img_raw = img.tobytes()

        example = tf.train.Example()
        feature = example.features.feature
        feature['img_data'].bytes_list.value.append(img_raw)
        feature['label'].int64_list.value.append(label)
        tf_writer.write(example.SerializeToString())

    tf_writer.close()


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
    label = tf.cast(features['label'], tf.int32)

    return img, label


# 该函数用于统计 TFRecord 文件中的样本数量(总数)
def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums


def train_data():
    batch_size = 10
    batch_num = total_sample('dogVScat.tfrecord') / batch_size
    print("batch_num:{}".format(batch_num))

    filename_queue = tf.train.string_input_producer(['dogVScat.tfrecord'], shuffle=False)
    image, label = read_and_decode_dogVScat(filename_queue)
    image_train, label_train = tf.train.batch([image, label], batch_size=batch_size, num_threads=1, capacity=32)
    # image_train, label_train = tf.train.shuffle_batch([image, label], batch_size, num_threads=1, capacity=5+batch_size*3, min_after_dequeue=5)
    # tf.one_hot 独热函数
    train_labels_one_hot = tf.one_hot(label_train, 2, on_value=1.0, off_value=0.0)
    x_data = tf.placeholder(tf.float32, shape=[None, 180, 180, 3])
    y_target = tf.placeholder(tf.float32, shape=[None, Class_Nums])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                  dtype=tf.int32)

    model_output = conv_net(x_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=model_output))
    optimizer = tf.train.AdamOptimizer(10e-5).minimize(loss, global_step=global_step)

    train_correct_prediction = tf.equal(tf.argmax(model_output, 1), tf.argmax(y_target, 1))
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
            print("xxxx:" + str(i))
            for j in range(int(batch_num)):
                print (1)
                image_batch, label_batch = session.run([image_train, train_labels_one_hot])
                _, step, acc, cost = session.run([optimizer, global_step, train_accuracy, loss],
                                                 feed_dict={x_data: image_batch, y_target: label_batch})
                acc_avg += (acc / batch_num)
                cost_avg += (cost / batch_num)
                print("acc_avg:{}".format(acc_avg))
                print("cost_avg:{}".format(cost_avg))
            print("step %d, training accuracy %0.10f loss %0.10f" % (i, acc_avg, cost_avg))
            loss_list.append(cost_avg)
            acc_list.append(acc_avg)
            saver.save(session, '/Users/closer/PycharmProjects/dogAndCat/model.ckpt', global_step=i)
    except tf.errors.OutOfRangeError:
        print('Done training --epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    session.close()

    plt.title('train loss')
    plt.plot(range(0, 100), loss_list, 'b-')
    plt.show()
    plt.title('accuracy')
    plt.plot(range(0, 100), acc_list, 'r--')
    plt.show()


def main():
    # gen_dogVScat_tfrecords('cat_vs_dog.tfrecord')
    train_data()
    # print (total_sample('cat_vs_dog.tfrecord'))


def test_one_image(image_path,model_path):
    image = Image.open(image_path)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [-1, 180, 180, 3])
    test_result = conv_net(image)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            print('no checkpoint file')
            return

    result = sess.run(test_result)
    result_label = np.argmax(result[0])
    print ("result:" + str(result_label))




if __name__ == '__main__':
    main()
