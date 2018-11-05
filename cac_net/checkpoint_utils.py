# -*- coding: utf-8 -*-
"""
Checkpoint utilities for cac_net

"""
import inspect
import sys
import os

import cac_net.constants


def get_checkpoint_dir(subdir=None):
    """ Creates a checkpoint directory from the filename of the calling
        program and returns it as its value. """
    if not subdir:
        current_path = inspect.getfile(sys._getframe(1))
        basename = os.path.basename(current_path)
        basename = os.path.splitext(basename)[0]
        subdir = basename
    checkpoint_dir = os.path.join(cac_net.constants.CHECKPOINT_DIR, subdir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        print('creating %s' % checkpoint_dir)
    return checkpoint_dir

def clean_up_checkpoints(checkpoint_dir=cac_net.constants.CHECKPOINT_DIR):
    """ Deletes all but the most recent hdf5 weight files from each checkpoint
        folder. Useful for quickly freeing up hard disk space taken up by
        all the model checkpoints.
    """
    for d in os.listdir(checkpoint_dir):
        directory = os.path.join(checkpoint_dir, d)
        files = os.listdir(directory)
        if len(files) == 0:
            print('removing %s' % directory)
            os.rmdir(directory)
        else:
            model_files = [os.path.join(directory, f) for f in files if f.endswith('.hdf5')]
            # sorts from oldest to most recent
            model_files.sort(key=lambda x: os.path.getmtime(x))
            for model_file in model_files[:-1]:
                print('removing %s' % model_file)
                os.remove(model_file)
