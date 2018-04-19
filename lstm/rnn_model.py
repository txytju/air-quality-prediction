import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from .lstm_data_util import generate_toy_data_for_lstm, generate_train_samples



def generate_train_samples(x, y, batch_size=32, num_periods=168, f_horizon=48):

    '''
    x, y are dataframe of shape (len_data, features)

    return:
        input_seq : shape of (batch_size, num_periods-f_horizon, feature_dim)
        output_seq : shape of (batch_size, f_horizon, feature_dim)
    '''
    total_start_points = len(x) - num_periods
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
    
    input_batch_idxs = [list(range(i, i+(num_periods-f_horizon))) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+(num_periods-f_horizon), i+num_periods)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq


def single_rnn_model(X_train, y_train, X_test, y_test, cell="BasicRNNCell", 
              learning_rate=0.001, epochs=500, print_every=100, 
              inputs=1, outputs=1, hidden=100, num_periods=20):
    '''
    X_tarin : x traning data, shape of [None, num_periods, inputs].
    y_train : y training data, shape of [None, num_periods, outputs].
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




def rnn_model(X_train, y_train, X_test, y_test, cell="BasicRNNCell", 
              learning_rate=0.001, epochs=500, print_every=100, 
              inputs=1, outputs=1, hidden=100, num_periods=168, f_horizon=48):
    '''
    X_tarin : x traning data, dataframe, shape of (length, features).
    y_train : y training data, dataframe, shape of (length, features).


    X_test : x data for prediction, shape of (length, features).
    Y_test : ground truth of y data when doing prediction, shape of (length, features).



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
            X_train_batch, Y_train_batch = generate_train_samples(X_train, y_train, batch_size=32, input_seq_len=num_periods-f_horizon, output_seq_len=f_horizon)
            _, loss_value = sess.run([training_op, loss], feed_dict={X:X_train_batch, y:Y_train_batch})
            losses.append(loss_value)
            
            if ep % print_every == 0:
                mse = loss.eval(feed_dict={X:X_train_batch, y:Y_train_batch})
                print(ep, "  MSE on training batch :", mse)

        
        y_pred = sess.run(outputs, feed_dict={X:X_test})
        # print(y_pred)

    plt.title("Forecast vs Actual", fontsize=14)
    plt.plot(pd.Series(np.ravel(y_test)), 'bo', markersize=10, label="Actual")
    plt.plot(pd.Series(np.ravel(y_pred)), 'r.', markersize=10, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")

    return losses, y_pred
