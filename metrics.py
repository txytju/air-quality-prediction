import numpy as np

def symmetric_mean_absolute_percentage_error(actual, forecast):
    '''
    Compute the Symmetric mean absolute percentage error (SMAPE or sMAPE) on the dev set or test set
    Details of SMAPE here : https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Args:
        actual : actual values in the dev/test dataset.
        forecast : model forecast values.

    '''
    actual = np.squeeze(actual)
    forecast = np.squeeze(forecast)

    assert len(actual) == len(forecast), "The shape of actual value and forecast value are not the same."

    length = len(actual)

    r = 0

    for i in range(length):
        f = forecast[i]
        a = actual[i]

        r += abs(f-a) / ((abs(a)+abs(f))/2)

    return r/length

