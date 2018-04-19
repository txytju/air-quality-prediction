import numpy as np

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