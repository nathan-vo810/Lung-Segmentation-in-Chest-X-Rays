import numpy as np
from keras import backend as K
import cv2
import os
import pickle


def evaluate(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    print(y_true_f)
    print(y_pred_f)

    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))

    possible_negatives = K.sum(K.round(K.clip((1 - y_true_f), 0, 1)))

    tp = K.eval(true_positives)
    pp = K.eval(possible_positives)
    pn = K.eval(possible_negatives)
    predict_pos = K.eval(predicted_positives)

    precision = tp / predict_pos
    recall = tp / pp
    f1 = 2 * precision * recall / (precision + recall)

    fp = predict_pos - tp
    fn = pp - tp
    tn = pn - fp

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / (pp + pn)
    IOU = tp / (pp + predict_pos - tp)

    print("TP: {}".format(tp))
    print("TN: {}".format(tn))
    print("FP: {}".format(fp))
    print("FN: {}".format(fn))
    print("Total: {}".format(total))
    print("Acc: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("IOU: {}".format(IOU))


def main():
    UNet_result = 'Unet_result_new'
    HybridNet_result = 'HybridNet_result'
    CNN_result = 'CNN_result'
    v1_result = 'v1-99'

    label_path = 'test_label/'

    images = os.listdir(label_path)
    images = [file for file in images if file.endswith('.jpg')]

    for img in images:
        print("V3 - Evaluating: {}".format(img))
        y_true = cv2.imread(os.path.join(label_path, img), 0).astype(np.float32)
        # height, width = y_true.shape[:2]
        y_pred = cv2.imread(os.path.join(v1_result, img), 0).astype(np.float32)
        # y_pred = cv2.resize(y_pred, (width, height)).astype(np.float32)
        evaluate(y_true, y_pred)


main()
