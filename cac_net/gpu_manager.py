# -*- coding: utf-8 -*-
"""
A fairly crude program for managing GPU resources, modeled on stanza. Better
approaches are now available.

"""
import random
import subprocess
import time
import os
import atexit

from cac_net.constants import GPU_TRACKER_DIR, N_GPUS


class GPUCountError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

def refresh_gpu_tracker(tracker_dir=GPU_TRACKER_DIR):
    # Make sure the directory exists
    if not os.path.exists(tracker_dir):
        os.mkdir(tracker_dir)
    gpus = set()
    utilizations = numbered_utilizations()
    max_retries = 300
    n_retries = 0
    while len(utilizations) != N_GPUS:
        utilizations = numbered_utilizations()
        n_retries += 1
        if n_retries > max_retries:
            time.sleep(1)
            msg = 'number of GPUs found (%s) does not match N_GPUS (%s) after %s retries' % (len(utilizations), N_GPUS, max_retries)
            raise GPUCountError(msg)
    remove_tracking_files(tracker_dir)
    available = available_gpus(utilizations)
    gpus = gpus.union(set(available))
    for gpu in gpus:
        mark_gpu_available(gpu)

def remove_tracking_files(tracker_dir):
    for file in os.listdir(tracker_dir):
        os.remove(os.path.join(tracker_dir, file))    

def mark_gpu_available(gpu, tracker_dir=GPU_TRACKER_DIR):
    """ Add tracking file indicating specified GPU is available """
    out_path = os.path.join(tracker_dir, gpu)
    f = open(out_path, 'w')
    f.close()

def mark_gpu_unavailable(gpu, tracker_dir=GPU_TRACKER_DIR):
    """ Remove a tracking file to indicate the specified GPU is not available """
    out_path = os.path.join(tracker_dir, gpu)
    os.remove(out_path)
    
def available_tracked_gpus(tracker_dir=GPU_TRACKER_DIR):
    return os.listdir(tracker_dir)

def set_tracked_gpu(gpu):
    mark_gpu_unavailable(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('CUDA_VISIBLE_DEVICES set to %s' % gpu)     
    atexit.register(mark_gpu_available, gpu)

def set_random_tracked_gpu(open_gpus):
    gpu = random.sample(open_gpus, 1)[0]
    set_tracked_gpu(gpu)

def set_cuda_visible_devices(wait_for_gpu=True):
    """ Finds an available GPU and sets the CUDA_VISIBLE_DEVICES variable
        to allow use of only that GPU. If wait_for_gpu=False it will default
        to CPU is no available GPU's are found.
    """
    open_gpus = available_tracked_gpus() # has to happen first so CPU mode can be set
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
        print('CUDA_VISIBLE_DEVICES has already been set to %s' % devices)
    else:
        if open_gpus:
            set_random_tracked_gpu(open_gpus)
        else:
            if wait_for_gpu and (N_GPUS > 0):
                while len(open_gpus) == 0:
                    print('no open GPUs found, waiting...')
                    time.sleep(10)
                    open_gpus = available_tracked_gpus()
                set_random_tracked_gpu(open_gpus)
            else:
                print('using CPU instead')
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'                

def nvidia_smi_output():
    try:
        proc = subprocess.Popen('nvidia-smi', stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        output, error = proc.communicate()
    except Exception:
        print("Couldn't run nvidia-smi, setting to CPU only mode")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        output=None
    return output
        
def gpu_utilizations(output):
    output = str(output)
    start = output.index('|===')
    end = output.index('\\n   ')
    lines = output[start:end].split('\\n')[2::3]
    fields = [line.split() for line in lines]
    #processing = [float(field[12].strip('%')) for field in fields]
    memory = [float(field[8].strip('MiB')) for field in fields]
    return memory

def numbered_utilizations():
    rows = []
    output = nvidia_smi_output()
    if output:
        utilizations = gpu_utilizations(output)
        for n, i in enumerate(utilizations):
            rows.append({'gpu': str(n), 'utilization': i})
    return rows

def available_gpus(utilizations=None):
    if not utilizations:
        utilizations = numbered_utilizations()
    return [r['gpu'] for r in utilizations if r['utilization'] == float(0)]

def unavailable_gpus(utilizations=None):
    if not utilizations:
        utilizations = numbered_utilizations()
    return [r['gpu'] for r in utilizations if r['utilization'] != float(0)]

def all_gpus(utilizations=None):
    if not utilizations:
        utilizations = numbered_utilizations()
    return [r['gpu'] for r in utilizations]       
