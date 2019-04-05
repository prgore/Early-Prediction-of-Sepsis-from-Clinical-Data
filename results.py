from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import metrics

import keras
import keras.backend as K

class Results(keras.callbacks.Callback):
    def __init__(self,val_X, val_y, batch_size, results):
        super(Results, self).__init__()
        self.val_X = val_X
        self.val_y = val_y
        self.batch_size=batch_size
        self.results = results

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on validation")
        yhat =self.model.predict(self.val_X, batch_size = self.batch_size)
        yhat = np.array(yhat)[:, 0]
        result = metrics.print_metrics_binary(np.array(self.val_y), yhat)
        self.results.append(result)