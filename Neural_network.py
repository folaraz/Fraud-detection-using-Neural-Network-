import pickle as pk
import numpy as np
import tensorflow as tf


d_file = open('credit_card.pickle', 'rb')
train_x, train_y, test_x, test_y = pk.load(d_file)
d_file.close()

batch_size = 100


node_hl1 = 400
node_hl2 = 400
node_hl3 = 400

n_class = 2

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layer_1 = {'Weights': tf.Variable(tf.random_normal([len(train_x[0]), node_hl1])),
                      'biases': tf.Variable(tf.random_normal([node_hl1]))}
    hidden_layer_2 = {'Weights': tf.Variable(tf.random_normal([node_hl1, node_hl2])),
                      'baises': tf.Variable(tf.random_normal([node_hl2]))}
    # hidden_layer_3 = {'Weights': tf.Variable(tf.random_normal([node_hl2, node_hl3])),
    #                   'baises': tf.Variable(tf.random_normal([node_hl3]))}
    output_layer = {'Weights': tf.Variable(tf.random_normal([node_hl3, n_class])),
                      'baises': tf.Variable(tf.random_normal([n_class]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1['Weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['Weights']), hidden_layer_2['baises'])
    l2 = tf.nn.relu(l2)

    # l3 = tf.add(tf.matmul(l2, hidden_layer_3['Weights']), hidden_layer_3['baises'])
    # l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l2, output_layer['Weights']), output_layer['baises'])

    return output


def train_neurals(x):
    predict = neural_network_model(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y))
    Optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    with tf.Session() as sess:
        initialization = tf.initialize_all_variables()
        sess.run(initialization)
        hm_epoch = 20

        for epoch in range(hm_epoch):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _ , c = sess.run([Optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch,'Completed out of', hm_epoch,'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neurals(x)