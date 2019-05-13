import tensorflow as tf
from tensorflow.data import TextLineDataset
import numpy as np
import sklearn
from data import feature
import os
import random
import PIL.Image as Image
from data import lwlrap


class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """
    def __init__(self):
        self.class_num = feature.class_num       # 输出类别的数目
        self.img_height = feature.img_height
        self.img_width = feature.img_width       # 图像的尺寸
        self.use_img_input = False        # 是否使用图片作为输入，False使用原始的feature vector
        self.dropout_keep_prob = 0.5     # dropout保留比例
        self.learning_rate = 5e-4   # 学习率
        self.batch_size = 128         # 批大小
        self.epoch_num = 300      # 总迭代轮次


class CNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.use_img_input = config.use_img_input
        self.batch_size = config.batch_size
        if self.use_img_input:
            self.input_x_dim = 3
        else:
            self.input_x_dim = 1

    def _set_input(self):
        # Input layer
        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.training = tf.placeholder(tf.bool, name='training')
        self.input_x = tf.reshape(self.next_element[0], [-1, self.img_height, self.img_width, self.input_x_dim], name='input_x')
        self.input_y = self.next_element[1]

    def _set_loss(self):
        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + tf.losses.get_regularization_loss()

    def _conv(self, input, ksize, stride, filters):
        return tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=[ksize, ksize],
            strides=[stride, stride],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.0),
        )

    def _conv_BN_relu(self, input, ksize, stride, filters, training):
        output = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=[ksize, ksize],
            strides=[stride, stride],
            padding='SAME',
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.0),
        )
        output = tf.layers.batch_normalization(output, training=training)
        return tf.nn.relu(output)

    def _maxpool_2x2(self, input):
        return tf.layers.max_pooling2d(
            inputs=input,
            pool_size=[2, 2],
            strides=[2, 2],
            padding='SAME',
        )

    def _avgpool_2x2(self, input):
        return tf.layers.average_pooling2d(
            inputs=input,
            pool_size=[2, 2],
            strides=[2, 2],
            padding='SAME'
        )

    def _fc(self, input, units, dropout_keep_prob, name=None):
        fc_output = tf.layers.dense(
            inputs=input,
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.0),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001),
            bias_regularizer = tf.contrib.layers.l2_regularizer(0.001),
            name=name
            )

        return tf.layers.dropout(fc_output, dropout_keep_prob)

    '''
    def convert_input(self, lines):
        """
        把读取的字符串数据转为形状为[batch_size, img_size, img_size, 1]的数组，
        作为CNN的输入

        :param pixels:
        :param labels:
        :return:
        """
        batch_size = lines.shape[0]
        batch_x = []
        batch_y = []

        for i in range(batch_size):
            line_ = lines[i].decode('utf-8').strip().split(',')
            fname = line_[0]
            if self.use_img_input:
                # 图片输入
                image = np.asarray(Image.open('data/' + feature.TRAIN_CURATED_IMAGE_DIR + '/' + fname),
                                   dtype=np.float32)
                batch_x.append(image)
                labels = [x.strip("\"") for x in line_[1:]]
            else:
                features = line_[1].split(' ')
                feature_vector = np.asarray([float(x) for x in features], dtype=np.float32)
                batch_x.append(feature_vector)
                labels = self.fname2label[fname[:8] + '.wav'].strip('\"').split(',')
            # 生成one_hot标签
            one_hot_label = np.zeros(shape=[self.class_num])
            for l in labels:
                one_hot_label[self.class2id[l]] = 1
            batch_y.append(one_hot_label)

        batch_x = np.stack(batch_x)
        batch_x = batch_x.reshape([batch_size, self.img_width, self.img_height, self.input_x_dim])
        batch_y = np.asarray(batch_y)

        return batch_x, batch_y
    '''
    def prepare_data(self):
        self.class2id = feature.class2id

        # 读取标签，转换成{fname, label}的键值对
        # ==================================================
        self.fname2label = {}
        with open('data/' + feature.TRAIN_CURATED_LABEL_PATH, 'r', encoding='utf-8') as f:
            f.readline()  # 跳过标题
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.strip()
                fname = line[:12]
                label = line[13:].strip("\"")
                self.fname2label[fname] = label
        # =======================================================================
        total_size = feature.TRAIN_CURATED_NON_SILENT_SIZE
        train_size = int(total_size * 0.7)

        print('Load and shuffle dataset...')

        # 读取图片和label的map函数
        def read_image(*row):
            image = tf.read_file('data/' + feature.TRAIN_CURATED_IMAGE_DIR + '/' + row[0])
            image = tf.image.decode_jpeg(image)
            image = tf.cast(image, tf.float32)
            label = tf.cast(row[1:], tf.float32)
            return image, label

        if self.use_img_input:
            path = 'data/' + feature.TRAIN_CURATED_IMAGE_LABEL_PATH
            print('Using image input.')
            print('Loading data from {}'.format(path))
            dataset = tf.data.experimental.CsvDataset(path,
                                                      record_defaults=[tf.string]+[tf.int32 for _ in range(self.class_num)],
                                                      header=True)
            dataset = dataset.map(read_image).shuffle(total_size)
        else:
            # 读取.npz文件
            path = 'data/' + feature.TRAIN_CURATED_NUMPY_PATH
            print('Using feature vector input.')
            print('Loading data from {}'.format(path))
            data = np.load(path)
            self.features = data['log_melgram']
            self.labels = data['labels']

            self.features_placeholder = tf.placeholder(tf.float32, self.features.shape, name='features')
            self.labels_placeholder = tf.placeholder(tf.float32, self.labels.shape, name='labels')
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.features_placeholder, self.labels_placeholder)).shuffle(total_size)

        train_dataset = dataset.take(train_size).batch(self.batch_size)
        valid_dataset = dataset.skip(train_size).batch(self.batch_size)
        # Create a reinitializable iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        train_init_op = iterator.make_initializer(train_dataset)
        valid_init_op = iterator.make_initializer(valid_dataset)

        self.next_element = iterator.get_next(name='next_element')

        return train_init_op, valid_init_op
        # ==============================================================

    def setVGG13(self):
        """
        在此函数中设定模型
        :return:
        """
        self._set_input()

        # conv3-64
        conv3_64_1 = self._conv_BN_relu(self.input_x, 3, 1, 64, self.training)
        conv3_64_output = self._conv_BN_relu(conv3_64_1, 3, 1, 64, self.training)

        # maxpool-1
        maxpool_1_output = self._maxpool_2x2(conv3_64_output)

        # conv3-128
        conv3_128_1 = self._conv_BN_relu(maxpool_1_output, 3, 1, 128, self.training)
        conv3_128_output = self._conv_BN_relu(conv3_128_1, 3, 1, 128, self.training)

        # maxpool-2
        maxpool_2_output = self._maxpool_2x2(conv3_128_output)

        # conv3-256
        conv3_256_1 = self._conv_BN_relu(maxpool_2_output, 3, 1, 256, self.training)
        conv3_256_output = self._conv_BN_relu(conv3_256_1, 3, 1, 256, self.training)

        # maxpool-3
        maxpool_3_output = self._maxpool_2x2(conv3_256_output)

        # conv3-512
        conv3_512_1 = self._conv_BN_relu(maxpool_3_output, 3, 1, 512, self.training)
        conv3_512_output = self._conv_BN_relu(conv3_512_1, 1, 1, 512, self.training)

        # maxpool-4
        maxpool_4_output = self._maxpool_2x2(conv3_512_output)

        # conv4-512
        conv3_512_1 = self._conv_BN_relu(maxpool_4_output, 3, 1, 512, self.training)
        conv3_512_output = self._conv_BN_relu(conv3_512_1, 3, 1, 512, self.training)

        # avgpool-5
        avgpool_5_output = self._avgpool_2x2(conv3_512_output)

        # flatten
        shape = avgpool_5_output.shape.as_list()
        dims = shape[1]*shape[2]*shape[3]
        maxpool_5_output_flatten = tf.reshape(avgpool_5_output, [-1, dims])

        # 输出层
        self.score = self._fc(maxpool_5_output_flatten, self.class_num, self.dropout_keep_prob, name='score')

        self.prediction = tf.nn.softmax(self.score, name='prediction')
        self._set_loss()





