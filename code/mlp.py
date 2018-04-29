from __future__ import print_function
from data_preprocessing_unsw import import_and_clean
import tensorflow as tf

#import dataset
train = import_and_clean("UNSW-NB15_1.csv")
test = import_and_clean("UNSW-NB15_2.csv")

#parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 80
display_step = 1

#network parameters
n_hidden_1 = 8
n_hidden_2 = 8
n_input = 2
n_classes = 2

#tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

#weights and biases
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}

#create model
def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#construct model
logits = multilayer_perceptron(X)

#define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        #loop over all batches
        for i in range(total_batch):
            batch_x = train.iloc[(i*batch_size):((i+1)*batch_size), [9, 36]]
            batch_y = train.iloc[(i*batch_size):((i+1)*batch_size), -1]
            #run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            #compute avg loss
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    print("Optimization finished")

    #test model
    pred = tf.nn.softmax(logits) # apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test.iloc[:, [9, 36]], Y: test.iloc[:, -1]}))
