import tensorflow.contrib.slim.python.slim.nets.vgg as vgg
import tensorflow as tf
from tensorflow.contrib import slim


class Net(object):
    def __init__(self, shape=(200,130), learning_rate=0.0001, keep_prob=1):
        self.learning_rate = learning_rate
        self.x = tf.placeholder('float', (None, shape[1], shape[0] ,1))
        self.y_ = tf.placeholder('float', (None, 2))
        self.keep_prob = keep_prob
        self.vgg16()
        self.get_loss()
        self.get_train_step()
        self.merged = tf.summary.merge_all()

    def weight_variable(self, shape, stddev=0.01):
        initial = tf.truncated_normal(shape, stddev=stddev, seed=123)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w, padding='VALID'):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(self, x, ksize=[1, 2, 2, 1]):
        return tf.nn.max_pool(x, ksize=ksize, strides=[1, 2, 2, 1], padding='SAME')

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            # 计算参数的均值，并使用tf.summary.scaler记录
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            tf.summary.histogram('histogram', var)

    def vgg16(self):
        with tf.name_scope('layer1'):
            with tf.name_scope('weights'):
                w_conv1 = self.weight_variable([3, 3, self.x.get_shape().as_list()[3], 8])
                self.variable_summaries(w_conv1)
            with tf.name_scope('biases'):
                b_conv1 = self.bias_variable([w_conv1.get_shape().as_list()[3]])
                self.variable_summaries(b_conv1)
            with tf.name_scope('conv_operation'):
                h_conv1 = tf.nn.relu(self.conv2d(self.x, w_conv1, padding='SAME') + b_conv1)
                tf.summary.histogram('Relu', h_conv1)

        with tf.name_scope('layer2'):
            with tf.name_scope('weights'):
                w_conv2 = self.weight_variable([3, 3, h_conv1.get_shape().as_list()[3], 16])
                self.variable_summaries(w_conv2)
            with tf.name_scope('biases'):
                b_conv2 = self.bias_variable([w_conv2.get_shape().as_list()[3]])
                self.variable_summaries(b_conv2)
            with tf.name_scope('conv_operation'):
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, padding='SAME') + b_conv2)
                tf.summary.histogram('Relu', h_conv1)

        with tf.name_scope('layer3'):
            with tf.name_scope('pool'):
                h_pool1 = self.max_pool_2x2(h_conv2, ksize=[1, 3, 3, 1])
                norm1 = tf.nn.lrn(h_pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
                tf.summary.histogram('lrn1', norm1)

        with tf.name_scope('layer4'):
            with tf.name_scope('weights'):
                w_conv3 = self.weight_variable([3, 3, norm1.get_shape().as_list()[3], 32])
                self.variable_summaries(w_conv3)
            with tf.name_scope('biases'):
                b_conv3 = self.bias_variable([w_conv3.get_shape().as_list()[3]])
                self.variable_summaries(b_conv3)
            with tf.name_scope('conv_operation'):
                h_conv3 = tf.nn.relu(self.conv2d(norm1, w_conv3, padding='SAME') + b_conv3)
                tf.summary.histogram('Relu', h_conv3)

        with tf.name_scope('layer5'):
            with tf.name_scope('weights'):
                w_conv4 = self.weight_variable([3, 3, h_conv3.get_shape().as_list()[3], 64])
                self.variable_summaries(w_conv4)
            with tf.name_scope('biases'):
                b_conv4 = self.bias_variable([w_conv4.get_shape().as_list()[3]])
                self.variable_summaries(b_conv4)
            with tf.name_scope('conv_operation'):
                h_conv4 = tf.nn.relu(self.conv2d(h_conv3, w_conv4, padding='SAME') + b_conv4)
                tf.summary.histogram('Relu', h_conv4)

        with tf.name_scope('layer6'):
            with tf.name_scope('pool'):
                h_pool2 = self.max_pool_2x2(h_conv4, ksize=[1, 3, 3, 1])
                norm2 = tf.nn.lrn(h_pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
                tf.summary.histogram('lrn2', norm2)
                tf.summary.histogram('h_pool2', h_pool2)

        with tf.name_scope('layer7'):
            with tf.name_scope('weights'):
                w_conv45 = self.weight_variable([3, 3, norm2.get_shape().as_list()[3], 128])
                self.variable_summaries(w_conv45)
            with tf.name_scope('biases'):
                b_conv45 = self.bias_variable([w_conv45.get_shape().as_list()[3]])
                self.variable_summaries(b_conv4)
            with tf.name_scope('conv_operation'):
                h_conv45 = tf.nn.relu(self.conv2d(norm2, w_conv45, padding='SAME') + b_conv45)
                tf.summary.histogram('Relu', h_conv45)

        with tf.name_scope('layer8'):
            dropout = tf.nn.dropout(h_conv45, keep_prob=self.keep_prob)

        with tf.name_scope('layer9'):
            with tf.name_scope('conn'):
                shape = dropout.get_shape().as_list()
                fc1 = tf.reshape(dropout, shape=[-1, shape[1] * shape[2] * shape[3]])
            with tf.name_scope('conv'):
                with tf.name_scope('weights'):
                    w_conv5 = self.weight_variable([fc1.get_shape().as_list()[1], 256], stddev=0.005)
                    self.variable_summaries(w_conv5)
                with tf.name_scope('biases'):
                    b_conv5 = self.bias_variable([w_conv5.get_shape().as_list()[1]])
                    self.variable_summaries(b_conv5)
                with tf.name_scope('conv_operation'):
                    h_conv5 = tf.nn.relu(tf.matmul(fc1, w_conv5) + b_conv5)
                    tf.summary.histogram('Relu', h_conv5)

        with tf.name_scope('layer10'):
            with tf.name_scope('weights'):
                w_conv6 = self.weight_variable([h_conv5.get_shape().as_list()[1], 128], stddev=0.005)
                self.variable_summaries(w_conv6)
            with tf.name_scope('biases'):
                b_conv6 = self.bias_variable([w_conv6.get_shape().as_list()[1]])
                self.variable_summaries(b_conv6)
            with tf.name_scope('conv_operation'):
                h_conv6 = tf.nn.relu(tf.matmul(h_conv5, w_conv6) + b_conv6)
                tf.summary.histogram('Relu', h_conv6)

        with tf.name_scope('layer11'):
            with tf.name_scope('weights'):
                w_conv7 = self.weight_variable([h_conv6.get_shape().as_list()[1], 2], stddev=0.005)
                self.variable_summaries(w_conv7)
            with tf.name_scope('biases'):
                b_conv7 = self.bias_variable([w_conv7.get_shape().as_list()[1]])
                self.variable_summaries(b_conv7)
            with tf.name_scope('conv_operation'):
                h_conv7 = tf.nn.softmax(tf.matmul(h_conv6, w_conv7) + b_conv7)
                tf.summary.histogram('Relu', h_conv7)

        self.predictions = h_conv7

    def get_loss(self):
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.predictions), axis=1))
        # self.loss = tf.reduce_mean(tf.square(self.predictions-self.y_))
        tf.summary.scalar('loss', self.loss)

    def get_train_step(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



if __name__ == '__main__':
    net = Net()
    vgg = net.predictions
    print()
