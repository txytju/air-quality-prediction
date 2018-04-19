import numpy as np

def symmetric_mean_absolute_percentage_error(actual, forecast, y_mean, y_std):
    '''
    Compute the Symmetric mean absolute percentage error (SMAPE or sMAPE) on a single data of the dev set or test set.
    Details of SMAPE here : https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Args:
        actual : an actual values in the dev/test dataset.
        forecast : an model forecast values.
        y_mean : mean value used when doing preprocess.
        y_std : std value used when doing preprocess.
        
    '''
    actual = np.squeeze(actual)
    forecast = np.squeeze(forecast)

    assert len(actual) == len(forecast), "The shape of actual value and forecast value are not the same."

    length = len(actual)
    actual = actual * y_std + y_mean
    forecast = forecast * y_std + y_mean

    r = 0

    for i in range(length):
        f = forecast[i]
        a = actual[i]

        r += abs(f-a) / ((abs(a)+abs(f))/2)

    return r/length


def SMAPE_on_dataset(actual_data, forecast_data, feature_list, y_mean, y_std, forecast_duration=24):
    '''
    Compute SMAPE value on the dataset of actual and forecast.
	
    Args:
        actual_data : actual data in the dev set or test set, shape [number_of_examples_in_the_dev_set, output_seq_length, num_of_output_features]
        forecast_data : forecast data which are predicted by the seq2seq model using x data in the dev set or test set, same shape as actual_data.
        forecast_duration : predict every 24 hours, so forecast_duration is set default to 24.
        feature_list : a list of features that is caculated in the forecast.
    Return:
        aver_smapes : average smape for all features on the test data.
        smapes_of_features : smapes of different features on the test data.
    '''
    assert actual_data.shape == forecast_data.shape, "The shape of actual data and perdiction data must match."

    number_of_features = actual_data.shape[2]
    smapes_list_of_features = {feature:[] for feature in feature_list}

    for i in range(0, actual_data.shape[0], forecast_duration):
        actual_data_item = actual_data[i]
        forecast_data_item = forecast_data[i]
        for j in range(number_of_features):
            feature = feature_list[j]
            a = actual_data_item[:,j]
            f = forecast_data_item[:,j]
            smape_a_feature_a_day = symmetric_mean_absolute_percentage_error(a, f, y_mean, y_std)
            smapes_list_of_features[feature].append(smape_a_feature_a_day)

    smapes_of_features = {feature:np.mean(value) for feature, value in smapes_list_of_features.items()}
    aver_smapes = np.mean(list(smapes_of_features.values()))

    return aver_smapes, smapes_of_features
    
        




