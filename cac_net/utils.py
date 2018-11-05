# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:35:08 2016

@author: MEASURE_A
"""
import os
import numpy as np
np.random.seed(1337)
import random
import time
import csv

import pandas as pd
from sklearn.externals import joblib

import cac_net.constants


CACHE_DATA_FORMAT = 'cached_{n_train}_{n_test}_{train_years}_{test_years}.csv'
CACHE_DIR = os.path.join(cac_net.constants.DATA_DIR, 'cache')
  
        
def data_source_as_df(data_source=cac_net.constants.DEFAULT_DATA_SOURCE):
    data_file = open(data_source, 'r', encoding='utf-8')
    df = pd.read_csv(data_file, na_filter=False, dtype=object)
    return df

def cache_fname(n_train, n_test, train_years, test_years):
    temp = 'cached_{n_train}_{n_test}_{tr_start}-{tr_stop}_{ts_start}-{ts_stop}'
    return temp.format(n_train=n_train, n_test=n_test, 
                       tr_start=train_years[0], tr_stop=train_years[1],
                       ts_start=test_years[0], ts_stop=test_years[1])

def cache_exists(folder_name):
    train_exists = os.path.exists(os.path.join(CACHE_DIR, folder_name, 'train.csv'))
    test_exists = os.path.exists(os.path.join(CACHE_DIR, folder_name, 'test.csv'))
    return train_exists and test_exists

def save_cache(train_rows, test_rows, folder_name):
    """ Create and cache a train/test dataset to speed up retrieval in the future. 
    """
    out_dir = os.path.join(CACHE_DIR, folder_name)
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    train_path = os.path.join(out_dir, 'train.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    for path, rows in zip([train_path, test_path], [train_rows, test_rows]):
        with open(path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=[col for col in rows[0]])
            writer.writeheader()
            writer.writerows(rows)
        print('%s rows written to %s' % (len(rows), path))

def load_cache(folder_name):
    """ load a train_rows, test_rows cache from disk """
    train_path = os.path.join(CACHE_DIR, folder_name, 'train.csv')
    test_path = os.path.join(CACHE_DIR, folder_name, 'test.csv')
    train_df = data_source_as_df(train_path)
    test_df = data_source_as_df(test_path)
    train_rows = train_df.to_dict(orient='records')
    test_rows = test_df.to_dict(orient='records')
    print('%d training rows loaded' % len(train_rows))
    print('%d testing rows loaded' % len(test_rows))
    return train_rows, test_rows

def get_train_test(n_train=None, n_test=None, 
                   train_years=(2011, 2014), 
                   test_years=(2015, 2015),
                   required_fields=cac_net.constants.CODE_TYPES_EXCEPT_SS, 
                   data_source=cac_net.constants.DEFAULT_DATA_SOURCE,
                   create_cache=True):
    """
    Retrieve examples for training and test. Optionally cache them for faster
    processing later on.
    """
    train_years_set = set(range(train_years[0], train_years[1] + 1))
    test_years_set = set(range(test_years[0], test_years[1] + 1))
    print('train years set to %s' % sorted(train_years_set))
    print('test years set to %s' % sorted(test_years_set))
    assert not test_years_set.issubset(train_years_set), 'test years should not be in train years'
    cache_folder = cache_fname(n_train=n_train, n_test=n_test, 
                             train_years=train_years, test_years=test_years)
    if cache_exists(cache_folder):
        print('loading cached data')
        train_rows, test_rows = load_cache(cache_folder)
    else: 
        print('retrieving rows from %s' % data_source)
        df = data_source_as_df(data_source=data_source)
        train_df = df[(df['survey_year'].astype(int) >= train_years[0]) & (df['survey_year'].astype(int) <= train_years[1])]
        test_df = df[(df['survey_year'].astype(int) >= test_years[0]) & (df['survey_year'].astype(int) <= test_years[1])]
        train_rows = train_df.to_dict(orient='records')
        test_rows = test_df.to_dict(orient='records')
        print('%d training rows loaded' % len(train_rows))
        print('%d testing rows loaded' % len(test_rows))
        # Randomize
        random.seed(31)
        random.shuffle(train_rows)
        random.shuffle(test_rows)
        train_rows = train_rows[: n_train]
        test_rows = test_rows[: n_test]
        print('%d rows retrieved for training' % len(train_rows))
        print('%d rows retrieved for testing' % len(test_rows))
        if create_cache:
            save_cache(train_rows=train_rows, test_rows=test_rows, folder_name=cache_folder)
    return train_rows, test_rows

def get_gold_data(data_source=os.path.join(cac_net.constants.DATA_DIR, 'n12_gold_1k_ud.csv')):
    with open(data_source, 'r', newline='', encoding='cp1252') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    print('%s rows retrieved' % len(rows))
    return rows

def get_verify_2016():
    data_source = os.path.join(cac_net.constants.DATA_DIR, 'verify_2016.csv')
    return get_gold_data(data_source)

def remove_partially_coded_csv(data_source):
    """ Removes cases without all 5 codes from a CSV file """
    out_rows = []
    n_read = 0
    n_removed = 0
    mandatory_fields = ['soc', 'nature_code', 'part_code', 'event_code', 'source_code']
    with open(data_source, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_read += 1
            if all([row[field].strip() for field in mandatory_fields]):
                out_rows.append(row)
            else:
                n_removed += 1
    print('%s rows read from %s' % (n_read, data_source))
    print('%s rows removed for missing codes' % n_removed)
    # shuffle out_rows
    random.seed(31)
    random.shuffle(out_rows)
    # write out_rows
    dirname, fname = os.path.split(data_source)
    out_fname = 'clean_%s' % fname
    out_path = os.path.join(dirname, out_fname)
    with open(out_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
        print('%s rows written to %s' % (len(out_rows), out_path))


def clean_and_shuffle_csv(data_source):
    """ Preprocesses the n12_2011-2015.csv file to remove any cases without
        all 5 major OIICS codes and also converts the entire file to utf-8 """
    # read in and filter data_source
    fields = ['soc', 'nature_code', 'part_code', 'event_code', 'source_code']
    out_rows = []
    n_read = 0
    n_removed = 0
    with open(data_source, 'r', newline='', encoding='cp1252') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_read += 1
            if all([row[field].strip() for field in fields]):
                out_rows.append(row)
            else:
                n_removed += 1
    print('%s rows read from %s' % (n_read, data_source))
    print('%s rows removed for missing codes' % n_removed)
    # shuffle out_rows
    random.seed(31)
    random.shuffle(out_rows)
    # write out_rows
    dirname, fname = os.path.split(data_source)
    out_fname = 'clean_%s' % fname
    out_path = os.path.join(dirname, out_fname)
    with open(out_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
        print('%s rows written to %s' % (len(out_rows), out_path))

def valid_codes(label_field, code_detail=None):
    """ Return the list of valid codes for a specific label field
    """
    type_map = {'socc' : 'soc',
                'part' : 'part_code',
                'even' : 'event_code',
                'sour' : 'source_code',
                'natu' : 'nature_code'}
    code_set = set()
    if label_field == 'sec_source_code':
        label_field = 'source_code'
        code_set.add('')
    df = pd.read_csv(os.path.join(cac_net.constants.DATA_DIR, 'code_set.csv'))
    rows = df.to_dict(orient='records')
    for row in rows:
        code_type = type_map[row['code_type']]
        code = row['code'][0: code_detail]
        if code_type == label_field:
            code_set.add(code)
    return list(code_set)

def balanced_weights(class_indexes, smoothing_factor=0, smoothing_constant=0):
    """ Creates a class_weight dictionary to use in fitting the model that
        weights each class by its inverse frequency so that all classes
        "occur" an equal number of times.
        
        smoothing_factor - portion of equal empirical sample added to each class,
            .5 implies that .5/1.5 of the count is class equal, 1/1.5 is empirical 
            1 implies that half the count is class equal and half is empirical
            2 implies that 2/3 of the count is class equal and 1/3 is empirical
        smoothing_constant - # of imaginary samples added to each class
    """
    n_samples = len(class_indexes)
    n_classes = len(set(class_indexes))
    counts = np.bincount(class_indexes)
    pseudo_counts = counts + (smoothing_factor * n_samples) / float(n_classes) + smoothing_constant
    pseudo_sample = sum(pseudo_counts)
    weights = pseudo_sample / (n_classes * pseudo_counts)
    index_weight = dict(zip(range(len(counts)), weights))
    return index_weight
    
def training_filepath(directory, n_train, labels, metrics_names=['acc', 'macro_f1']):
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(metrics_names, str):
        metrics_names = [metrics_names]
    scores = []
    for metrics_name in metrics_names:
        scores.append(metrics_name)
        for label in labels:
            if len(labels) == 1:
                prefix = ''
                metric_str = '{val_%s:.3f}' % metrics_name
            else:
                prefix = label[0].upper()
                metric_str = '{val_%s_%s:.3f}' % (label, metrics_name)
            scores.append('{prefix}{metric_str}'.format(prefix=prefix, metric_str=metric_str))
    scores_str = '_'.join(scores)
    filename = 'model_%dK.{epoch:02d}-%s.hdf5' % ((n_train/1000.), scores_str)
    filepath = os.path.join(directory, filename)
    return filepath

def get_most_recent_file_name(dir_name, extension=''):
    """ For the specified directory, return the name of the most recently
        modified file with the specified extension.
    """
    files = [f for f in os.listdir(dir_name) if os.path.splitext(f)[-1] == extension]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_name, x)),
               reverse=True)
    file_name = files[0]
    print('loading most recent %s file: %s' % (extension, file_name))
    return file_name

def get_most_recent_files(dir_name, extension=''):
    files = [f for f in os.listdir(dir_name) if os.path.splitext(f)[-1] == extension]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_name, x)),
               reverse=True)    
    return files

def jload(dir_name, file_name=None):
    directory = os.path.join(cac_net.constants.CHECKPOINT_DIR, dir_name)
    if not file_name:
        file_name = get_most_recent_file_name(directory, extension='')
    path = os.path.join(directory, file_name)
    return joblib.load(path)

def jdump(classifier, dir_name=None, file_name=None):
    """ Saves a full classifier, with vectorizer and keras model"""
    if not file_name:
        file_name = time.strftime('%Y-%m-%d')
    if not dir_name:
        dir_name = cac_net.constants.CHECKPOINT_DIR
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    out_path = os.path.join(dir_name, file_name)
    joblib.dump(classifier, out_path)
    
def load_from_checkpoint(dir_name, file_name=None, wait_for_gpu=True, 
                         custom_objects=None):
    """ Loads only the Keras model (not vectorizers) from a checkpoint directory. """
    from cac_net.gpu_manager import set_cuda_visible_devices
    set_cuda_visible_devices(wait_for_gpu=wait_for_gpu)
    from keras.models import load_model
    import h5py
    
    directory = os.path.join(cac_net.constants.CHECKPOINT_DIR, dir_name)
    if not file_name:
        file_name = get_most_recent_file_name(directory, extension='.hdf5')
    path = os.path.join(directory, file_name)
    # HACK to fix keras optimizer weights loading error
    f = h5py.File(path, 'r+')
    try:
        del f['optimizer_weights']
    except KeyError:
        pass
    # END HACK
    return load_model(path, custom_objects=custom_objects)

class Classifier:
    def __init__(self, meta_vectorizer, model_path, preprocessor=None, custom_objects=None):
        self.meta_vectorizer = meta_vectorizer
        self.model_path = model_path
        self.custom_objects = custom_objects
        if preprocessor:
            self.preprocessor = preprocessor
        
    def load(self, dir_name, wait_for_gpu=True, custom_objects=None):
        self.model = load_from_checkpoint(dir_name, wait_for_gpu=wait_for_gpu, 
                                          custom_objects=custom_objects)

def load_clf(dir_name, file_name=None, wait_for_gpu=True, custom_objects=None):
    """ Load classifier, complete with metavectorizer and keras model """
    clf = jload(dir_name, file_name)
    clf.load(dir_name, wait_for_gpu=wait_for_gpu, custom_objects=custom_objects)
    return clf
