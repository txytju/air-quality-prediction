import numpy as np
import matplotlib.pyplot as plt

def generate_toy_data_for_lstm(num_periods = 120, f_horizon = 4, samples = 10020):
    '''
    Generate toy data.
    Args:
        num_periods : total length of time series.
        f_horizon : number of prediction value, so overlapings are (num_periods - f_horizon)
        samples : total number of x series.
    '''
    # data  : t*sin(t)/3 + 2*sin(5*t)
    t = np.linspace(0,100,num=samples)
    ts = t*np.sin(t)/3 + 2.*np.sin(5.*t)
    plt.plot(t,ts);
    
    TS = np.array(ts)

    x_data = TS[:(len(TS)-(len(TS) % num_periods))]
    y_data = TS[f_horizon : (len(TS)-(len(TS) % num_periods)+f_horizon)]
    print("length of training data x : ", x_data.shape)
    print("length of training data y : ", y_data.shape)

    x_batches = x_data.reshape(-1,num_periods,1)
    y_batches = y_data.reshape(-1,num_periods,1)

    print("training data x shape : ", x_batches.shape)
    
    test_x_setup = TS[-(num_periods + f_horizon):]
    testX = test_x_setup[:num_periods].reshape(-1,num_periods,1) 
    testY = TS[-(num_periods):].reshape(-1,num_periods,1)
    
    return x_batches, y_batches, testX, testY


def generate_data_for_lstm(ts, num_periods = 120, f_horizon = 4):
    '''
    Generate data for single variable lstm model. 
    Args:
        ts : time series to be used.
        num_periods : total length of time series.
        f_horizon : number of prediction value, so overlapings are (num_periods - f_horizon)
    Return:
        
    '''
    
    TS = np.array(ts)

    x_data = TS[:(len(TS)-(len(TS) % num_periods))]
    y_data = TS[f_horizon : (len(TS)-(len(TS) % num_periods)+f_horizon)]
    print("length of training data x : ", x_data.shape)
    print("length of training data y : ", y_data.shape)

    x_batches = x_data.reshape(-1,num_periods,1)
    y_batches = y_data.reshape(-1,num_periods,1)

    print("training data x shape : ", x_batches.shape)
    
    test_x_setup = TS[-(num_periods + f_horizon):]
    testX = test_x_setup[:num_periods].reshape(-1,num_periods,1) 
    testY = TS[-(num_periods):].reshape(-1,num_periods,1)
    
    print("test data x shape : ", testX.shape)
    
    return x_batches, y_batches, testX, testY


# For multi variable lstm model

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


def generate_test_samples(x, y, num_periods=168, f_horizon=48):
    
    total_samples = x.shape[0]
    
    input_batch_idxs = [list(range(i, i+(num_periods-f_horizon))) for i in range((total_samples-num_periods))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [list(range(i+(num_periods-f_horizon), i+num_periods)) for i in range((total_samples-num_periods))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    
    return input_seq, output_seq

