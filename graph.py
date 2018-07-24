import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import cutfeature as cf
import tensorflow.contrib.slim as slim
batch_size = 8

Re = 0.00000012
START = int(len(cf.train_data)/3)
END = len(cf.train_data)


# w1 = tf.Variable(tf.random_normal([201, 30], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([30, 4], stddev=1, seed=1))

x = tf.placeholder(tf.float32,shape=(None, 240), name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None, 4), name='y-input')


dataset_size = len(cf.train_data[START:END])
print('dataset_size',dataset_size)
X = np.array(cf.train_data[START:END])
Y = cf.Y_date[START:END]
print('X:',X)
print('Y:',Y)
X_test = np.array(cf.train_data[0:START])
Y_test = cf.Y_date[0:START]


net1 = slim.fully_connected(x, 600, activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(Re))
net1 = slim.fully_connected(net1, 300, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(Re))
net1 = slim.fully_connected(net1, 70, activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(Re))
net1 = slim.fully_connected(net1, 4, activation_fn=None, weights_regularizer=slim.l2_regularizer(Re))

tf.losses.softmax_cross_entropy(y_, net1)
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

#   定义损失函数和反向传播算法
# cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(total_loss)

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()#tf.initialize_all_variables()
#    初始化变量
    sess.run(init_op)

 #   print(sess.run(w1))
#    print(sess.run(w2))
#    print(sess.run(y,feed_dict={x: X}))

#   设定训练的轮数
    STEPS = 50000
    for i in range(STEPS):
        #   每次取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        #   通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entrocp = sess.run(total_loss, feed_dict={x: X, y_: Y})
            print("After %d tranning steps,cross entrocp on all date is %g" % (i, total_cross_entrocp))

    result = sess.run(net1, feed_dict={x: X_test})
    print(type(result))

    def jiance():
        s = np.zeros(4)
        p = np.zeros(4)   # 保存错误模式
        for ly in Y_test:
            for i in range(len(ly)):
                s[i] += ly[i]
        print('s:', s)

        for i in range(result.shape[0]):
            # t = 0
            # for j in range(result.shape[1]):
            #   if result[i][j]>result[i][t]:
            #       t=j
            t = np.argmax(result[i])
            q = Y_test[i].index(1)
            if q != t:  # 出错
                p[q] += 1
            #    print('wrong!!!!q==%d t==%d' % (q, t), Y_test[i], result[i])
        print('总准确率：',np.true_divide(sum(s)-sum(p),sum(s)),'\n各类别准确率:',np.true_divide(s-p,s))
    jiance()
