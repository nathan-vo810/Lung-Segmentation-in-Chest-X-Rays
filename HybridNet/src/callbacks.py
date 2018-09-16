import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        cm1 = confusion_matrix(val_targ, val_predict)

        tn = np.diag(cm1)[0]
        fn = np.diag(np.fliplr(cm1))[1]
        tp = np.diag(cm1)[1]
        fp = np.diag(np.fliplr(cm1))[0]

        _val_precision = tp / (tp + fp)
        _val_recall = tp / (tp + fn)
        _val_f1 = 2 * (_val_recall * _val_precision) / (_val_recall + _val_precision)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return
