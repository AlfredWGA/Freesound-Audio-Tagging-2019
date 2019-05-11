# coding=utf-8
import sklearn.metrics as metrics
import sklearn as sk
import tensorflow as tf
from cnn_model import CNN
from cnn_model import CNNConfig
import datetime
import time
import os
import numpy as np
from data import lwlrap


def train():
    # Training procedure
    # ======================================================
    # 设定最小显存使用量
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        config = CNNConfig()
        cnn = CNN(config)
        cnn.setVGG13()

        print('Setting Tensorboard and Saver...')
        # 设置Saver和checkpoint来保存模型
        # ===================================================
        checkpoint_dir = os.path.join(os.path.abspath("checkpoints"), "cnn")
        checkpoint_prefix = os.path.join(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # =====================================================

        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        # ====================================================================
        train_tensorboard_dir = 'tensorboard/cnn/train/'
        valid_tensorboard_dir = 'tensorboard/cnn/valid/'
        if not os.path.exists(train_tensorboard_dir):
            os.makedirs(train_tensorboard_dir)
        if not os.path.exists(valid_tensorboard_dir):
            os.makedirs(valid_tensorboard_dir)

        # 训练结果记录
        log_file = open(valid_tensorboard_dir+'/log.csv', mode='w', encoding='utf-8')
        log_file.write(','.join(['epoch', 'loss', 'precision', 'recall', 'f1_score']) + '\n')

        merged_summary = tf.summary.merge([tf.summary.scalar('loss', cnn.loss)])

        train_summary_writer = tf.summary.FileWriter(train_tensorboard_dir, sess.graph)
        # =========================================================================

        global_step = tf.Variable(0, trainable=False)
        # 衰减的学习率，每1000次衰减4%
        learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                   global_step, decay_steps=5000, decay_rate=0.98, staircase=False)

        # 保证Batch normalization的执行
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
            # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cnn.loss, global_step)
            train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(cnn.loss, global_step)

        # 训练步骤
        def train_step(batch_x, batch_y, keep_prob=config.dropout_keep_prob):
            feed_dict = {
                cnn.input_x: batch_x,
                cnn.input_y: batch_y,
                cnn.dropout_keep_prob: keep_prob,
                cnn.training: True
            }
            sess.run(train_op, feed_dict=feed_dict)
            feed_dict = {cnn.input_x: batch_x,
                cnn.input_y: batch_y,
                cnn.dropout_keep_prob: 1.0,
                cnn.training: False}
            step, loss, y_pred, summery = sess.run(
                [global_step, cnn.loss, cnn.prediction, merged_summary],
                feed_dict=feed_dict)
            lrap = lwlrap.calculate_overall_lwlrap_sklearn(truth=batch_y, scores=y_pred)
            # per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth=batch_y, scores=y_pred)
            # mean_lwlrap = np.sum(per_class_lwlrap * weight_per_class)
            t = datetime.datetime.now().strftime('%m-%d %H:%M')
            print('%s: epoch: %d, step: %d, loss: %f, lwlrap: %f' % (t, epoch, step, loss, lrap))
            # 把结果写入Tensorboard中
            train_summary_writer.add_summary(summery, step)

        # 验证步骤
        def valid_step(next_valid_element):
            # 把valid_loss和valid_accuracy归0
            y_true = []
            y_pred = []
            valid_loss = 0.0
            i = 0
            while True:
                try:
                    lines = sess.run(next_valid_element)
                    batch_x, batch_y = cnn.convert_input(lines)
                    feed_dict = {
                        cnn.input_x: batch_x,
                        cnn.input_y: batch_y,
                        cnn.dropout_keep_prob: 1.0,
                        cnn.training: False
                    }
                    # loss, pred, true = sess.run([cnn.loss, cnn.prediction, cnn.labels], feed_dict)
                    # 多次验证，取loss和prediction均值
                    mean_loss = 0
                    mean_pred = 0
                    for i in range(config.multi_valid_num):
                        loss, pred = sess.run([cnn.loss, cnn.prediction], feed_dict)
                        mean_loss += loss
                        mean_pred += pred
                    mean_loss /= config.multi_valid_num
                    mean_pred /= config.multi_valid_num
                    y_pred.extend(mean_pred)
                    y_true.extend(batch_y)
                    valid_loss += mean_loss
                    i += 1
                except tf.errors.OutOfRangeError:
                    # 遍历完验证集，计算评估
                    valid_loss /= i
                    y_true = np.asarray(y_true)
                    y_pred = np.asarray(y_pred)
                    lrap = lwlrap.calculate_overall_lwlrap_sklearn(truth=y_true, scores=y_pred)
                    t = datetime.datetime.now().strftime('%m-%d %H:%M')
                    log = '%s: epoch %d, validation loss: %0.6f, lwlrap: %0.6f' % (t, epoch, valid_loss, lrap)
                    print(log)
                    log_file.write(log + '\n')
                    time.sleep(3)
                    return

        print('Start training CNN...')
        sess.run(tf.global_variables_initializer())
        train_init_op, valid_init_op, next_train_element, next_valid_element = cnn.prepare_data()
        # Training loop
        for epoch in range(config.epoch_num):
            sess.run(train_init_op)
            while True:
                try:
                    lines = sess.run(next_train_element)
                    batch_x, batch_y = cnn.convert_input(lines)
                    train_step(batch_x, batch_y, config.dropout_keep_prob)
                except tf.errors.OutOfRangeError:
                    # 初始化验证集迭代器
                    sess.run(valid_init_op)
                    # 计算验证集准确率
                    valid_step(next_valid_element)
                    break
        train_summary_writer.close()
        log_file.close()
        # 训练完成后保存参数
        path = saver.save(sess, checkpoint_prefix, global_step=global_step)
        print("Saved model checkpoint to {}\n".format(path))
    # ==================================================================


if __name__ == '__main__':
    train()






