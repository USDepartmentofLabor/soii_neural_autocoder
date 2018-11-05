# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:45:25 2017

@author: MEASURE_A
"""
import numpy as np
import math
import collections


def get_indexed_rows(rows, code_field):
    indexed_rows = collections.defaultdict(list)
    for row in rows:
        code = row[code_field]
        indexed_rows[code].append(row)
    return indexed_rows

def even_sample(indexed_rows, code_field, n_codes, total_sample_size):
    """ data - n_example length array representing training data
    """
    subsample_size = math.ceil(total_sample_size / float(n_codes))
    equal_indexed_rows = {}
    for code, rows in indexed_rows.items():
        n_rows = len(rows)
        rows = rows.copy()
        rows = np.random.permutation(rows)
        if n_rows < subsample_size:
            n_repeats = math.ceil(subsample_size / float(n_rows))
            rows = np.repeat(rows, n_repeats)
        equal_indexed_rows[code] = list(rows[0: subsample_size])
    sampled_rows = []
    while len(sampled_rows) < total_sample_size:
        random_codes = np.random.permutation([c for c in equal_indexed_rows])
        for code in random_codes:
            row = equal_indexed_rows[code].pop()
            sampled_rows.append(row)
    sampled_rows = np.asarray(sampled_rows)
    return sampled_rows[0: total_sample_size]

def max_sample(rows, code_field, max_n):
    """ Resample rows so that no code occurs more than max_n times. """
    counts = collections.defaultdict(int)
    out_rows = []
    for row in rows:
        code = row[code_field]
        counts[code] += 1
        if counts[code] <= max_n:
            out_rows.append(row)
    print('rows reduced from %s to %s after max sampling' % (len(rows), len(out_rows)))
    return out_rows