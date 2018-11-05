# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:13:33 2017

@author: MEASURE_A
"""
import math
import random
import numpy as np
from keras import backend as K

def recall(y_true, y_pred):
    """Code specific recall"""
    true_positives = K.sum(y_true * y_pred, axis=0)
    possible_positives = K.sum(y_true, axis=0)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Code specific precision"""
    true_positives = K.sum(y_true * y_pred, axis=0)
    predicted_positives = K.sum(y_pred, axis=0)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    """Code specific f1"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f = 2*((p*r)/(p+r+K.epsilon()))
    return f

def macro_f1(y_true, y_pred):
    f = f1(y_true, y_pred) # [n_labels, 1]
    true_labels = true_label_exists(y_true)
    true_label_count = n_true_labels(true_labels)
    mean = K.sum(f) / true_label_count
    return mean    

def true_label_exists(y_true):
    return K.any(y_true, axis=0)

def n_true_labels(true_labels_present):
    true = K.cast(true_labels_present, 'float32')
    return K.sum(true)

def f_sum(f):
    return K.sum(f)

def accuracy(y_true, y_pred):
    true_positives = K.sum(y_true * y_pred)
    total = K.sum(y_true)
    return true_positives / total


def gen_data(n_rows=200, n_cols=10, forced_agreement=.5):
    n_true_missing = random.randint(0, n_cols-1)
    n_pred_missing = random.randint(0, n_cols-1)
    missing_true = random.sample(range(n_cols), n_true_missing)
    missing_pred = random.sample(range(n_cols), n_pred_missing)
    all_labels = set(range(n_cols))
    true_labels = all_labels - set(missing_true)
    pred_labels = all_labels - set(missing_pred)
    true = np.zeros((n_rows, n_cols))
    pred = np.zeros((n_rows, n_cols))
    for n in range(n_rows):
        true_label = random.sample(true_labels, 1)
        if random.random() < forced_agreement:
            pred_label = true_label
        else:
            pred_label = random.sample(pred_labels, 1)
        true[n, true_label] = 1
        pred[n, pred_label] = 1
    return true, pred


def test(print_results=False, **kwargs):
    import tensorflow as tf
    from sklearn.metrics import f1_score as sk_f1_score
    from sklearn.metrics import accuracy_score as sk_ac_score
    from sklearn.metrics import precision_recall_fscore_support
    
    ny_true, ny_pred = gen_data(**kwargs)
    y_pred = K.variable(ny_pred)
    y_true = K.variable(ny_true)
    
    y_true_exists = np.any(ny_true, axis=0)
    y_true_labels = np.argwhere(y_true_exists==True)
    y_true_labels = [i[0] for i in y_true_labels]

    sk_p, sk_r, sk_f, _ = precision_recall_fscore_support(ny_true, ny_pred)
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        tf_r = r.eval()
        tf_p = p.eval()
        tf_mf = mf1.eval()
        sk_mf = sk_f1_score(y_true=np.argmax(ny_true,axis=1), 
                            y_pred=np.argmax(ny_pred,axis=1), 
                            labels=y_true_labels, average='macro')
        #assert math.isclose(tf_r, sk_r, abs_tol=0.001)
        #assert math.isclose(tf_p, sk_p, abs_tol=0.001)
        assert math.isclose(tf_mf, sk_mf, abs_tol=0.001)
        if print_results:
            print('recall   ', tf_r)
            print('sk recall', sk_r)
            print('precision', tf_p)
            print('sk precis', sk_p)
            print('f        ', f1(y_true, y_pred).eval())
            print('sk f     ', sk_f)
            print('np: true_labels', y_true_labels, len(y_true_labels))
            print('tf: true_labels', true_label_exists(y_true).eval(), 
                                     n_true_labels(true_label_exists(y_true)).eval())
            print('macro f1 ', tf_mf, sk_mf)
            print('accuracy ', acc.eval(), sk_ac_score(ny_true, ny_pred))