"""
Holds constants used throughout the package
"""
import os
import socket
from cac_net.config import config


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'experiments')
IN_PROGRESS_DIR = os.path.join(EXPERIMENT_DIR, 'in_progress')
COMPLETED_EXPERIMENT_DIR = os.path.join(EXPERIMENT_DIR, 'completed') 
FAILED_EXPERIMENT_DIR = os.path.join(EXPERIMENT_DIR, 'error')
GPU_TRACKER_DIR = os.path.join(EXPERIMENT_DIR, 'gpu_tracker')

DEFAULT_DATA_SOURCE = os.path.join(DATA_DIR, 'clean_n12_2011-2016.csv')

if socket.gethostname() == config['GPU_HOST']:
    CHECKPOINT_DIR = r'/data/checkpoints'
    EMBEDDING_DIR = r'/data/embeddings'
    TRACKING_DB_PATH = r'/data/cac_data/oiics_review/code_review_tracker.db'
    TF_HUB = r'/data/tfhub'
    N_GPUS = 4
else:
    CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
    EMBEDDING_DIR = os.path.join(DATA_DIR, 'embeddings')
    TRACKING_DB_PATH = os.path.join(DATA_DIR, 'oiics_review', 'code_review_tracker.db')
    N_GPUS = 0

NAR_TYPES = ['nar_activity', 'nar_event', 'nar_source', 'nar_nature']
CODE_TYPES_EXCEPT_SS = ['soc', 'nature_code', 'part_code', 'event_code', 'source_code']
CODE_TYPES = CODE_TYPES_EXCEPT_SS + ['sec_source_code']
JOB_CATEGORIES = ['office', 'sales', 'assembly', 'repair', 'construction',
                  'health', 'driving', 'food', 'maintenance',
                  'material_handling', 'farming', 'other']