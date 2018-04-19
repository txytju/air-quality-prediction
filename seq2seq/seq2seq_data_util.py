import numpy as np

# For single varuable seq2seq model 
def generate_train_dev_set(ts, dev_set_proportion):
    '''
    args:
        ts : pandas timeseries
        dev_set_proportion : proportion of dev set in the data set.
    '''
    ts = ts.values
    all_length = len(ts)
    dev_length = int(dev_set_proportion * all_length)
    dev = ts[-dev_length:]
    train = ts[:-dev_length]
    
    return train, dev

def generate_training_data_for_seq2seq(ts, batch_size=10, input_seq_len=120, output_seq_len=48):
    '''
    Random generate training data with batch_size.
    args:
        ts : training time series to be used.
        batch_size : batch_size for the training data.
        input_seq_len : length of input_seq to the encoder.
        output_seq_len : length of output_seq of the decoder.
    returns:
        np.array(input_seq_x) shape : [batch_size, input_seq_len]
        np.array(output_seq_y) shape : [batch_size, output_seq_len]
    '''
    # TS = np.array(ts)
    TS = ts

    total_start_points = len(TS) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)
    
    input_seq = [TS[i:(i+input_seq_len)] for i in start_x_idx]
    output_seq = [TS[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]

    return np.array(input_seq), np.array(output_seq)

def generate_dev_data_for_seq2seq(ts, input_seq_len=120, output_seq_len=48):
    
    TS = ts
    dev_set = []
    total_start_points = len(TS) - input_seq_len - output_seq_len

    for i in range(total_start_points):
        input_seq = TS[i:(i+input_seq_len)]
        output_seq = TS[(i+input_seq_len):(i+input_seq_len+output_seq_len)]
        dev_set.append((input_seq, output_seq))

    return dev_set


def generate_x_y_data(ts, past_seq_length, future_sequence_length, batch_size):
    """
    Generate single feature data for seq2seq. Random choose batch_size data.
    
    args:
        ts is single feature time series. ts can be training data or validation data or test data.
        past_seq_length is seq_length of past data.
        future_sequence_length is sequence_length of future data.
        batch_size.

    returns: tuple (X, Y)
        X is (past_seq_length, batch_size, input_dim)
        Y is (future_sequence_length, batch_size, output_dim)

    """
    series = ts.values
    
    batch_x = []
    batch_y = []
    
    for _ in range(batch_size):
        
        total_series_num = len(series) - (past_seq_length + future_sequence_length)
        random_index = int(np.random.choice(total_series_num, 1))
        
        x_ = series[random_index : random_index + past_seq_length]
        y_ = series[random_index + past_seq_length : random_index + past_seq_length + future_sequence_length]


        batch_x.append(x_)
        batch_y.append(y_)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    batch_x = np.expand_dims(batch_x, axis=2)
    batch_y = np.expand_dims(batch_y, axis=2)
    # shape: (batch_size, seq_length, input/output_dim)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, input/output_dim)

    return batch_x, batch_y





# For multi variable seq2seq model

def generate_train_samples(x, y, batch_size=32, input_seq_len=30, output_seq_len=5):

    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)

def generate_test_samples(x, y, input_seq_len=30, output_seq_len=5):
    
    total_samples = x.shape[0]
    
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq