# coding=utf-8
import sklearn as sk
from data import feature
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import numpy as np


class Predictor(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def init_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.config = CNNConfig()
        # self.cnn = CNN(self.config)

        print('Loading model from file:', self.model_path)
        saver = tf.train.import_meta_graph(self.model_path + '.meta')
        saver.restore(self.sess, self.model_path)
        self.graph = tf.get_default_graph()
        # 从图中读取变量
        self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
        self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        self.prediction = self.graph.get_operation_by_name("prediction").outputs[0]
        self.training = self.graph.get_operation_by_name("training").outputs[0]

    def predict(self, batch_x):
        feed_dict = {
            self.input_x: batch_x,
            self.dropout_keep_prob: 1.0,
            self.training: False
        }
        pred = self.sess.run(self.prediction, feed_dict)
        return pred

    def tag_test_dataset(self, result_path):
        f = open(result_path, 'w', encoding='utf-8', newline='')
        writer = csv.writer(f)
        writer.writerow(['fname']+feature.classes)

        path = 'data/' + feature.TEST_NUMPY_PATH
        print('Using feature vector input.')
        print('Loading data from {}'.format(path))
        data = np.load(path)
        features = data['log_melgram']
        fnames = data['fnames']

        features_placeholder = tf.placeholder(tf.float32, features.shape)
        fnames_placeholder = tf.placeholder(tf.string, fnames.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, fnames_placeholder)).batch(128)

        # Create a reinitializable iterator
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        train_init_op = iterator.make_initializer(dataset)
        next_element = iterator.get_next()

        self.sess.run(train_init_op, feed_dict={features_placeholder: features, fnames_placeholder: fnames})

        # key是文件名，value是[score，times]的二元组
        scores = {}

        while True:
            try:
                batch = self.sess.run(next_element)
                features = np.reshape(batch[0], [-1, feature.img_height, feature.img_width, 1])
                fnames = batch[1]
                pred = self.predict(features)
                for i in range(len(fnames)):
                    fname = fnames[i].decode('utf-8')[:8] + '.wav'
                    if fname not in scores:
                        scores[fname] = [np.zeros(shape=[feature.class_num]), 0]
                    scores[fname][0] += pred[i]
                    scores[fname][1] += 1
            except tf.errors.OutOfRangeError:
                sf = open('data/' + feature.SAMPLE_PATH, 'r')
                reader = csv.reader(sf)
                next(reader)
                for line in reader:
                    fname = line[0]
                    scores[fname][0] /= scores[fname][1]
                    writer.writerow([fname]+scores[fname][0].tolist())
                sf.close()
                break
        f.close()


if __name__ == '__main__':
    predictor = Predictor('./checkpoints/cnn-28200')
    predictor.init_model()
    predictor.tag_test_dataset('submission.csv')

