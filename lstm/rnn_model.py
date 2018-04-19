import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

def rnn_model(X_train, y_train, X_test, y_test, cell="BasicRNNCell", 
              learning_rate=0.001, epochs=500, print_every=100, 
              inputs=1, outputs=1, hidden=100, num_periods=20):
    '''
    X_tarin : x traning data.
    y_train : y training data.
    X_test : x data for prediction.
    Y_test : ground truth of y data when doing prediction.
    cell : which kind of rnn cell to use, "BasicRNNCell" for BasicRNNCell, "LSTMCell" for LSTMCell
    learning_rate
    inputs : input_shape
    outputs : output_shape
    hidden : number of hidden units
    num_periods : length of time sequences.
    '''
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, num_periods, inputs])
    y = tf.placeholder(tf.float32, [None, num_periods, outputs])

    if cell == "BasicRNNCell" :
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
    elif cell == "LSTMCell" :
        basic_cell = tf.contrib.rnn.LSTMCell(num_units=hidden, activation=tf.nn.relu)
    else :
        print("Cell model wrong.")
        return

    rnn_output, status = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
    stacked_outputs = tf.layers.dense(stacked_rnn_output, outputs)
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, outputs])

    loss = tf.reduce_sum(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    losses = []
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epochs):
            _, loss_value = sess.run([training_op, loss], feed_dict={X:X_train, y:y_train})
            losses.append(loss_value)
            
            if ep % print_every == 0:
                mse = loss.eval(feed_dict={X:X_train, y:y_train})
                print(ep, "  MSE:", mse)

        y_pred = sess.run(outputs, feed_dict={X:X_test})
        # print(y_pred)

    plt.title("Forecast vs Actual", fontsize=14)
    plt.plot(pd.Series(np.ravel(y_test)), 'bo', markersize=10, label="Actual")
    plt.plot(pd.Series(np.ravel(y_pred)), 'r.', markersize=10, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")

    return losses, y_pred
