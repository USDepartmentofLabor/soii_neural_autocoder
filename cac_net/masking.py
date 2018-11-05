# -*- coding: utf-8 -*-
"""
Functions to assist with mask management (removing padding tokens).

@author: MEASURE_A
"""

from keras import backend as K
from keras.layers import Lambda
from keras.layers.merge import multiply


def mask_from_field_indicator_seq(field_sequence, start_index, end_index):
    """ Used for single sequence inputs. Can be used to mask OIICS narratives
        from SOC output layers, and vice versa. 
        start_index = field index of first unmasked field (should be a slice index)
        end_index = field index of last unmasked field (should be a slice index)
    """
    # field_sequence is [n_batch, n_steps, n_fields]
    # fields are indicated a value of 1 if field is present 0 otherwise
    # max will therefore be 1 if a field between start_index and end_index is present
    mask = Lambda(lambda x: K.max(x[:, :, start_index: end_index], axis=-1))(field_sequence)
    return mask

def mask_from_embedded_seq(input_sequence):
    # [n_batch, n_steps, n_hidden]
    # identify the batch steps with 0.0 as the first hidden_value
    bool_mask = Lambda(lambda x: K.not_equal(x[:,:,0], 0.0))(input_sequence)
    # bool_mask is [n_batch, n_steps] boolean matrix
    float_mask = Lambda(lambda x: K.cast(x, K.floatx()))(bool_mask)
    return float_mask

def masked_seq(input_sequence, mask):
    """ input_sequence - [n_batch, n_steps, n_hidden]
        mask - [n_batch, n_steps]
        output - [n_batch, n_steps, n_hidden] with masked steps zeroed out
    """
    # repeat the 1's and 0's along a new dimension (at position 1)
    repeated_mask = Lambda(K_repeat, arguments={'n_hidden': input_sequence.shape[-1].value})(mask)
    # reshufles the mask so that the repeated values are moved to the last dimension
    permuted_mask = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(repeated_mask)
    masked_sequence = multiply([input_sequence, permuted_mask])
    return masked_sequence
    
# possibly unnecessary but I did this so I could save the models, wasn't working otherwise
def K_repeat(x, n_hidden):
    return K.repeat(x, n_hidden)