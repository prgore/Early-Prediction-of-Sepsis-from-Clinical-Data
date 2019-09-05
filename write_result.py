import os
import numpy as np
def write_result(predicted_probability, thredhold = 0.5):
    name_columns = ['PredictedProbability', 'PredictedLabel']
    predicted_label = predicted_probability
    predicted_label[predicted_label>=thredhold] =1
    predicted_label[predicted_label<thredhold] =0
    frame = list
    frame
    # print(predicted_probability)
    return predicted_probability, predicted_label
x = np.array([0.3, 0.4, 0.5, 0.6])
print(x)
a,b = write_result(x, 0.5)
print(a)
print(b)