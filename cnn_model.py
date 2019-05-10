import tensorflow as tf
from tensorflow.data import TextLineDataset
import numpy as np
import sklearn
from data import feature
import os
import random


class CNNConfig(object):
    """
    # TODO: 在此修改TextCNN以及训练的参数
    """
    def __init__(self):
        self.class_num = feature.class_num       # 输出类别的数目
        self.img_width = feature.img_width       # 图像的尺寸
        self.img_heigth = feature.img_height
        self.dropout_keep_prob = 0.5     # dropout保留比例
        self.learning_rate = 1e-4   # 学习率
        self.multi_valid_num = 1   # 验证时，验证n次，结果取平均
        self.train_batch_size = 128         # 每批训练大小
        self.valid_batch_size = 500        # 每批测试大小
        self.valid_per_batch = 250           # 每多少批进行一次测试
        self.epoch_num = 300       # 总迭代轮次


class CNN(object):
    def __init__(self, config):
        self.class_num = config.class_num
        self.img_width = config.img_width
        self.img_height = feature.img_height
        self.train_batch_size = config.train_batch_size
        self.valid_batch_size = config.valid_batch_size
        self.valid_per_batch = config.valid_per_batch

    def _set_input(self):
        # Input layer
        self.input_x = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.class_num], name="labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 训练时batch_normalization的Training参数应为True,
        # 验证或测试时应为False
        self.training = tf.placeholder(tf.bool, name='training')

        # self.input_x_enhanced = tf.map_fn(self.image_enhance, self.input_x)
        self.input_x_enhanced = self.input_x

    def _set_loss(self):
        # Loss function
        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + tf.losses.get_regularization_loss()

        # Calculate accurac1y
        # with tf.name_scope('accuracy'):
        #     self.correct_predictions = tf.equal(tf.round(self.prediction), self.input_y)
        #     # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        #     self.all_labels_true = tf.reduce_min(tf.cast(self.correct_predictions, tf.float32), 1)
        #     self.accuracy = tf.reduce_mean(self.all_labels_true)

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

    def _maxpool_2x2(self, input):
        return tf.layers.max_pooling2d(
            inputs=input,
            pool_size=[2, 2],
            strides=[2, 2],
            padding='SAME',
        )

    def _fc(self, input, units, dropout_keep_prob, name=None):
        fc_output = tf.layers.dense(
            inputs=input,
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.initializers.constant(0.0),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
            bias_regularizer=tf.contrib.layers.l2_regularizer(0.001),
            name=name
            )
        return tf.layers.dropout(fc_output, dropout_keep_prob)

    def setVGG19(self):
        """
        在此函数中设定模型
        :return:
        """
        self._set_input()
        # 加入批标准化以减少过拟合
        input_x_norm = tf.layers.batch_normalization(self.input_x_enhanced, training=self.training)

        # conv3-64
        conv3_64_1 = self._conv(input_x_norm, 3, 1, 64)
        conv3_64_output = self._conv(conv3_64_1, 3, 1, 64)

        # maxpool-1
        maxpool_1_output = self._maxpool_2x2(conv3_64_output)

        # conv3-128
        conv3_128_1 = self._conv(maxpool_1_output, 3, 1, 128)
        conv3_128_output = self._conv(conv3_128_1, 3, 1, 128)

        # maxpool-2
        maxpool_2_output = self._maxpool_2x2(conv3_128_output)

        # conv3-256
        conv3_256_1 = self._conv(maxpool_2_output, 3, 1, 256)
        conv3_256_2 = self._conv(conv3_256_1, 3, 1, 256)
        conv3_256_3 = self._conv(conv3_256_2, 3, 1, 256)
        conv3_256_output = self._conv(conv3_256_3, 3, 1, 256)

        # maxpool-3
        maxpool_3_output = self._maxpool_2x2(conv3_256_output)

        # conv3-512
        conv3_512_1 = self._conv(maxpool_3_output, 3, 1, 512)
        conv3_512_2 = self._conv(conv3_512_1, 3, 1, 512)
        conv3_512_3 = self._conv(conv3_512_2, 3, 1, 512)
        conv3_512_output = self._conv(conv3_512_3, 3, 1, 512)

        # maxpool-4
        maxpool_4_output = self._maxpool_2x2(conv3_512_output)

        # conv4-512
        conv3_512_1 = self._conv(maxpool_4_output, 3, 1, 512)
        conv3_512_2 = self._conv(conv3_512_1, 3, 1, 512)
        conv3_512_3 = self._conv(conv3_512_2, 3, 1, 512)
        conv3_512_output = self._conv(conv3_512_3, 3, 1, 512)

        # maxpool-5
        maxpool_5_output = self._maxpool_2x2(conv3_512_output)

        # flatten
        shape = maxpool_5_output.shape.as_list()
        dims = shape[1]*shape[2]*shape[3]
        maxpool_5_output_flatten = tf.reshape(maxpool_5_output, [-1, dims])

        # fully-connected-1
        fc_1 = self._fc(maxpool_5_output_flatten, 2048, self.dropout_keep_prob)

        # fully-connected-2
        fc_2 = self._fc(fc_1, 2048, self.dropout_keep_prob)

        # fully-connected-3
        fc_3 = self._fc(fc_2, 1000, self.dropout_keep_prob)

        # 输出层
        self.score = self._fc(fc_3, self.class_num, self.dropout_keep_prob, name='score')

        self.prediction = tf.argmax(self.score, 1, name='prediction')

        # self._set_output()

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
            feature = line_[1].split()  # 像素值
            batch_x.append([float(x) for x in feature])
            # 把label转换成one_hot
            label = self.labels[fname[:8]+'.wav']
            batch_y.append(label)
        batch_x = np.stack(batch_x)
        batch_x = sklearn.preprocessing.normalize(batch_x)
        batch_x = batch_x.reshape([batch_size, self.img_width, self.img_height, 1])
        batch_y = np.asarray(batch_y)

        return batch_x, batch_y

    def prepare_data(self):
        # 读取标签，转换成{fname, one_hot_label}的键值对
        self.labels = {}
        with open('data/'+feature.TRAIN_CURATED_LABEL_PATH, 'r', encoding='utf-8') as f:
            f.readline()    # 跳过标题
            while True:
                line = f.readline()
                if line == '':
                    break
                line_ = line.strip().split(',')
                fname = line_[0]
                # 若标签多于1个，会用双引号括起来
                label = [l.strip("\"") for l in line_[1:]]
                one_hot_label = np.zeros(shape=[feature.class_num])
                for l in label:
                    one_hot_label[feature.class2id[l]] = 1
                self.labels[fname] = one_hot_label

        train_size = int(feature.TRAIN_CURATED_NON_SILENT_SIZE*0.7)
        self.train_dataset = TextLineDataset('data/' + feature.TRAIN_CURATED_NON_SILENT_PATH).skip(1).take(train_size)
        self.valid_dataset = self.train_dataset.skip(train_size)
        print('Shuffling dataset...')
        # 打乱数据
        train_dataset = self.train_dataset.shuffle(5000).batch(self.train_batch_size)
        valid_dataset = self.valid_dataset.shuffle(2500).batch(self.valid_batch_size)

        # Create a reinitializable iterator
        train_iterator = train_dataset.make_initializable_iterator()
        valid_iterator = valid_dataset.make_initializable_iterator()

        train_init_op = train_iterator.initializer
        valid_init_op = valid_iterator.initializer

        # 要获取元素，先sess.run(train_init_op)初始化迭代器
        # 再sess.run(next_train_element)
        next_train_element = train_iterator.get_next()
        next_valid_element = valid_iterator.get_next()

        return train_init_op, valid_init_op, next_train_element, next_valid_element
        # ==============================================================

    def image_enhance(self, image):
        """
        对图片进行数据增强
        :param image: shape为[heigth, width, 1]的numpy数组或tensor
        :return:
        """
        # 进行随机裁剪
        images_crop = tf.image.random_crop(image, [self.crop_size, self.crop_size, 1])

        # 随机水平翻转
        images_crop = tf.image.random_flip_left_right(images_crop)
        # 随机对比度
        images_crop = tf.image.random_contrast(images_crop, 0.5, 1.5)
        # 随机亮度
        images_crop = tf.image.random_brightness(images_crop, max_delta=0.5)

        noise = tf.random_normal(shape=tf.shape(images_crop), mean=0.0, stddev=10.0,
                                 dtype=tf.float32)
        images_crop = tf.add(images_crop, noise)

        return images_crop

    def setVGG16(self):
        """
        在此函数中设定模型
        :return:
        """
        self._set_input()
        # 加入批标准化以减少过拟合
        input_x_norm = tf.layers.batch_normalization(self.input_x_enhanced, training=self.training)

        # conv3-64
        conv3_64_1 = self._conv(input_x_norm, 3, 1, 64)
        conv3_64_output = self._conv(conv3_64_1, 3, 1, 64)

        # maxpool-1
        maxpool_1_output = self._maxpool_2x2(conv3_64_output)

        # conv3-128
        conv3_128_1 = self._conv(maxpool_1_output, 3, 1, 128)
        conv3_128_output = self._conv(conv3_128_1, 3, 1, 128)

        # maxpool-2
        maxpool_2_output = self._maxpool_2x2(conv3_128_output)

        # conv3-256
        conv3_256_1 = self._conv(maxpool_2_output, 3, 1, 256)
        conv3_256_2 = self._conv(conv3_256_1, 3, 1, 256)
        conv3_256_output = self._conv(conv3_256_2, 1, 1, 256)

        # maxpool-3
        maxpool_3_output = self._maxpool_2x2(conv3_256_output)

        # conv3-512
        conv3_512_1 = self._conv(maxpool_3_output, 3, 1, 512)
        conv3_512_2 = self._conv(conv3_512_1, 3, 1, 512)
        conv3_512_output = self._conv(conv3_512_2, 1, 1, 512)

        # maxpool-4
        maxpool_4_output = self._maxpool_2x2(conv3_512_output)

        # conv4-512
        conv3_512_1 = self._conv(maxpool_4_output, 3, 1, 512)
        conv3_512_2 = self._conv(conv3_512_1, 3, 1, 512)
        conv3_512_output = self._conv(conv3_512_2, 1, 1, 512)

        # maxpool-5
        maxpool_5_output = self._maxpool_2x2(conv3_512_output)

        # flatten
        shape = maxpool_5_output.shape.as_list()
        dims = shape[1]*shape[2]*shape[3]
        maxpool_5_output_flatten = tf.reshape(maxpool_5_output, [-1, dims])

        # fully-connected-1
        fc_1 = self._fc(maxpool_5_output_flatten, 2048, self.dropout_keep_prob)

        # fully-connected-2
        fc_2 = self._fc(fc_1, 2048, self.dropout_keep_prob)

        # fully-connected-3
        fc_3 = self._fc(fc_2, 1000, self.dropout_keep_prob)

        # 输出层
        self.score = self._fc(fc_3, self.class_num, self.dropout_keep_prob, name='score')

        self.prediction = tf.nn.sigmoid(self.score, name='prediction')
        self._set_loss()


