########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
import sys
import data_processing

data = data_processing.read_cifar10_data()
data.normalize()

class vgg16:
    def __init__(self, weights=None, sess=None, lr=5e-5, epochs=100, batch=100, decay=0.7, keep_rate=0.85):
        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        self.batch = batch
        self.keep_rate = keep_rate
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_drop_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        # # zero-mean input
        # # todo - this probably should be changed b/c it doesn't apply to our new dataset
        # with tf.name_scope('preprocess') as scope:
        #     resized = tf.image.resize_images(self.x, [224, 224])

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            out = tf.layers.batch_normalization(out, training=self.training)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 1024], dtype=tf.float32, stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            fc1l = tf.layers.batch_normalization(fc1l, training=self.training)
            self.fc1 = tf.nn.relu(fc1l)
            self.fc1 = tf.layers.dropout(self.fc1, 1-self.keep_drop_prob, training=self.training)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            fc2l = tf.layers.batch_normalization(fc2l, training=self.training)
            self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = tf.layers.dropout(self.fc2, 1-self.keep_drop_prob, training=self.training)
            self.parameters += [fc2w, fc2b]

        # fc3
        # the output of this layer must be changed in order to accommodate the fact that cifar-10 only has 10 labels
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.fc3l = tf.layers.batch_normalization(out, training=self.training)
            self.parameters += [fc3w, fc3b]

        out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.fc3l))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(out)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

    # code adapted from Liu Liu's LeNet implementation
    def train(self, report_freq=100):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        lr_l = []
        train_result = []
        valid_result = []
        stop_acc = 0
        stuck = 0

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        for i in range(self.epochs):
            sys.stdout.write("epoch {}: ".format(i))
            sys.stdout.flush()
            # todo - will have to implement batching for cifar-10
            batch = data.next_batch(self.batch)
            old_batch = []

            j = 0
            while batch[0] != []:
                if j%10 == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()
                self.sess.run([self.train_step, extra_update_ops], feed_dict={self.x: batch[0], self.y_: batch[1], self.training : True, self.keep_drop_prob : self.keep_rate})

                old_batch = batch
                batch = data.next_batch(self.batch)
                j += 1

            sys.stdout.write('\n')
            sys.stdout.flush()
            if i % report_freq == 0 and old_batch != []:

                train_acc = self.sess.run(self.accuracy,
                                          feed_dict={self.x: data.train_X[:10000], self.y_: data.train_y[:10000], self.training : False, self.keep_drop_prob : 1})
                train_result.append(train_acc)

                valid_acc = self.valid_eval()
                valid_result.append(valid_acc)

                lr_l.append(self.lr)

                stuck = stuck if valid_acc > stop_acc else stuck + 1
                stop_acc = valid_acc if valid_acc > stop_acc else stop_acc

                print('step %d, training accuracy %g' % (i, train_acc))
                print('step %d, validation accuracy %g' % (i, valid_acc))
                print('step %d, learning rate %g' % (i, self.lr))

            # learning rate decay
            if stuck >= 3 and stop_acc:
                self.lr *= self.decay
                stuck = 0

            if len(valid_result) > 6 and stop_acc not in valid_result[-6:]:
                break

        return train_result, valid_result, lr_l

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()

        # for i in range(0,10000,50):
        # for i in range(0, 500, 50):
        ave = self.sess.run(self.accuracy, feed_dict={self.x: data.test_X, self.y_: data.test_y, self.training : False, self.keep_drop_prob : 1})

        # ave = np.array(average).mean()

        print('Test accuracy accuracy %g' % (ave))

        return ave

    def valid_eval(self):
        self.eval()

        # for i in range(0,10000,50):
        # for i in range(0, 500, 50):
        ave = self.sess.run(self.accuracy, feed_dict={self.x: data.valid_X, self.y_: data.valid_y, self.training : False, self.keep_drop_prob : 1})
        # print(ave)

        # ave = np.array(average).mean()

        return ave

if __name__ == '__main__':
    print("initializing network")
    sess = tf.Session()
    tf.reset_default_graph()
    net = vgg16(sess=sess)
    print("training")
    train_acc, valid_acc, lr_l = net.train(1)
    print("testing")
    test_acc = net.test_eval()

    with open("scratchnet2.csv", 'w') as o:
        buffer = ','.join(["epoch"] + [str(i) for i in range(20)])+'\n'
        o.write(buffer)

        buffer = ','.join(["training"] + [str(i) for i in train_acc])+'\n'
        o.write(buffer)

        buffer = ','.join(["validation"] + [str(i) for i in valid_acc])+'\n'
        o.write(buffer)

        buffer = ','.join(["learning_rate"] + [str(i) for i in lr_l]) + '\n'
        o.write(buffer)

        buffer = str(test_acc)+'\n'
        o.write(buffer)
        o.flush()

    # sess = tf.Session()
    # imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    #
    # img1 = imread('laska.png', mode='RGB')
    # img1 = imresize(img1, (224, 224))
    #
    # prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print class_names[p], prob[p]
