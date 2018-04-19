import numpy as np
import matplotlib.pyplot as plt

def generate_toy_data_for_lstm(num_periods = 120, f_horizon = 4, samples = 10020):
    '''
    Generate toy data.
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
    ts : time series to be used.
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
    
    return x_batches, y_batches, testX, testY
