# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 14:42:10 2017

@author: MEASURE_A
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras.layers import TimeDistributed, Reshape, Dense, Activation, LSTM, Lambda
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Highway
from keras.layers import add, BatchNormalization
from keras.layers.merge import concatenate, multiply
from keras.layers.wrappers import Bidirectional
from keras.models import Model


class Attention(Layer):
    '''Attention operation for temporal data.
    # Input shape
    3D tensor with shape: `(samples, steps, features)`.
    # Output shape
    2D tensor with shape: `(samples, features)`.
    '''
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.glorot_uniform()
        self.attention_dim = attention_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((self.attention_dim, self.attention_dim),
                           name='{}_W'.format(self.name))
        self.b = K.zeros((self.attention_dim,), name='{}_b'.format(self.name))
        self.u = self.init((self.attention_dim,),
                           name='{}_u'.format(self.name))
        self.trainable_weights = [self.W, self.b, self.u]
        self.built = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        # Calculate the first hidden activations
        a1 = K.tanh(K.dot(x, self.W) + self.b)  # [n_samples, n_steps, n_hidden]
        # K.dot won't let us dot a 3D with a 1D so we do it with mult + sum
        mul_a1_u = a1 * self.u                  # [n_samples, n_steps, n_hidden]
        dot_a1_u = K.sum(mul_a1_u, axis=2)      # [n_samples, n_steps]
        # Calculate the per step attention weights
        a2 = K.softmax(dot_a1_u)
        a2 = K.expand_dims(a2)                  # [n_samples, n_steps, 1] so div broadcasts
        # Apply attention weights to steps
        weighted_input = x * a2                 # [n_samples, n_steps, n_features]
        # Sum across the weighted steps to get the pooled activations
        return K.sum(weighted_input, axis=1)
         
    
class MaskSeq(Layer):
    ''' Compute a sequence mask from an raw text input for a target hidden size.
        This mask will later be applied to the target input.
    '''
    def __init__(self, target_hidden, **kwargs):
        self.target_hidden = target_hidden
        super(MaskSeq, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.target_hidden)

    def call(self, text_input, mask=None):
        """ text_input should be a [None, max_words, max_chars] tensor.
        """
        bool_mask = K.not_equal(text_input[:,:,0], 0.0)         # [None, max_words]
        float_mask = K.cast(bool_mask, K.floatx())              # [None, max_words]
        repeat_mask = K.repeat(float_mask, self.target_hidden)  # [None, target_hidden, max_words]
        permute_mask = K.permute_dimensions(repeat_mask, [0, 2, 1]) # [None, max_words, target_hidden]
        return permute_mask
    
def K_mul(x):
    """ x should be a (weights, input) tuple
        weights should be [n_samples, n_steps]
        input should be [n_samples, n_steps, n_hidden]
        This function multiplies the inputs by the weights but does not sum.
        Used for multihead attention as described in "Attention is all you need".
    """
    weights = x[0]
    input = x[1]
    weights_3d = K.expand_dims(weights)     # [n_samples, n_steps, 1] so mult broadcasts
    weighted_input = input * weights_3d     # [n_samples, n_steps, n_features]
    return weighted_input

def get_K_mul_shape(input_shape):
    """ Input shape will contain the tuple of input shapes. We use the shape
        of the second input as the first is expanded to that size. """
    return input_shape[0]

def K_dot(x):
    """ weights should be [n_samples, n_steps]
        x should be [n_samples, n_steps, n_hidden]
        With Theano you can accomplish this with the merge([weights,x], mode=dot), 
        however this doesn't work in Tensorflow. This special function fixes
        this defficiency.
    """
    weights = x[0]
    input = x[1]
    weights_3d = K.expand_dims(weights)     # [n_samples, n_steps, 1] so mult broadcasts
    weighted_input = input * weights_3d     # [n_samples, n_steps, n_features]
    return K.sum(weighted_input, axis=1)    # [n_samples, n_features]

def get_K_dot_shape(input_shape):
    return (None, input_shape[1][2])

def masked_softmax(x):
    input_layer = x[0]
    custom_mask = x[1]
    num = K.exp(input_layer - K.max(input_layer, axis=-1, keepdims=True))
    masked_num = num * custom_mask
    masked_den = K.sum(masked_num, axis=-1, keepdims=True)
    return masked_num / masked_den

def masked_softmax_shape(input_shape):
    return input_shape

def mlp_attention(input, seq_len, n_hidden):
    h1 = TimeDistributed(Dense(n_hidden, activation='tanh'))(input)
    h2 = TimeDistributed(Dense(1, use_bias=False))(h1)     # [n_sample, n_steps, 1]
    flat = Reshape((seq_len,))(h2)                     # [n_sample, n_steps]
    attention = Activation('softmax')(flat)            # [n_sample, n_steps]
    attended_texts = Lambda(K_dot, output_shape=get_K_dot_shape)([attention, input])
    return attended_texts

def named_mlp_attention(input, seq_len, n_hidden, prefix, mask=None):
    h1 = TimeDistributed(Dense(n_hidden, activation='tanh'), 
                         name='%s_att_TDdense1' % prefix)(input)
    h2 = TimeDistributed(Dense(1, use_bias=False),
                         name='%s_att_TDdense2' % prefix)(h1)   # [n_sample, n_steps, 1]
    # calculate an importance "value" for each step
    flat = Reshape((seq_len,), name='%s_att_flat' % prefix)(h2) # [n_sample, n_steps]
    # convert the importance values to probabilities with the softmax
    attention = Activation('softmax', name='%s_att_softmax' % prefix)(flat) # [n_sample, n_steps]
    if mask is not None:
        # zero out the masked attention
        masked_attention = multiply([attention, mask], name='%s_masked_attention' % prefix)
        # renormalize the attention to a probability
        attention = Lambda(lambda x: x / K.sum(x, axis=1, keepdims=True), 
                           name='%s_masked_normed_attention' % prefix)(masked_attention)
    # weight each step by it's probabilistic importance and sum them together
    attended_texts = Lambda(K_dot, output_shape=get_K_dot_shape,
                            name='%s_summarized_attention' % prefix)([attention, input])
    return attended_texts

def dynamic_attention(input_layer, prefix, mask=None):
    n_hidden = input_layer.shape[-1].value
    h1 = TimeDistributed(Dense(n_hidden, activation='tanh'), 
                         name='%s_att_TDdense1' % prefix)(input_layer)
    h2 = TimeDistributed(Dense(1, use_bias=False),
                         name='%s_att_TDdense2' % prefix)(h1)   # [n_sample, n_steps, 1]
    # calculate an importance "value" for each step
    flat = Lambda(lambda x: K.squeeze(x, axis=-1), name='%s_att_flat' % prefix)(h2)
    # convert the importance values to probabilities with the softmax
    attention = Activation('softmax', name='%s_att_softmax' % prefix)(flat) # [n_sample, n_steps]
    if mask is not None:
        # zero out the masked attention
        masked_attention = multiply([attention, mask], name='%s_masked_attention' % prefix)
        # renormalize the attention to a probability
        attention = Lambda(lambda x: x / K.sum(x, axis=1, keepdims=True), 
                           name='%s_masked_normed_attention' % prefix)(masked_attention)
    # weight each step by it's probabilistic importance and sum them together
    attended_texts = Lambda(K_dot, output_shape=get_K_dot_shape,
                            name='%s_summarized_attention' % prefix)([attention, input_layer])
    return attended_texts


def feature_mlp_attention(input, seq_len, n_hidden, prefix, mask=None):
    """ Instead of per step attention, per step per feature attention.
        DiSAN https://arxiv.org/pdf/1709.04696v1.pdf
    """ 
    h1 = TimeDistributed(Dense(n_hidden, activation='tanh'), 
                         name='%s_att_TDdense1' % prefix)(input)
    h2 = TimeDistributed(Dense(n_hidden, activation='softmax'),
                         name='%s_att_softmax' % prefix)(h1)   # [n_sample, n_steps, n_hidden]
    attended_texts = multiply([h2, input], name='%s_attended' % prefix) # [n_sample, n_steps, n_hidden]
    summed_texts = Lambda(lambda x: K.mean(x, axis=1))(attended_texts)
    return summed_texts

def custom_mlp_attention(input, seq_len, n_hidden, prefix, mask=None):
    """ Doesn't work with current Tensorflow because of a _thread.lock pickle bug,
        but upgrading should fix that """
    h1 = TimeDistributed(Dense(n_hidden, activation='tanh'), 
                         name='%s_att_TDdense1' % prefix)(input)
    h2 = TimeDistributed(Dense(1, use_bias=False),
                         name='%s_att_TDdense2' % prefix)(h1)   # [n_sample, n_steps, 1]
    flat = Reshape((seq_len,), name='%s_att_flat' % prefix)(h2) # [n_sample, n_steps]
    if mask is not None:
        attention = Lambda(masked_softmax, output_shape=masked_softmax_shape, 
                           name='%s_att_softmax' % prefix)([flat, mask])        
    else:
        attention = Activation('softmax', name='%s_att_softmax' % prefix)(flat) # [n_sample, n_steps]
    attended_texts = Lambda(K_dot, output_shape=get_K_dot_shape)([attention, input])
    return attended_texts

def mlp_raw_attention(input, seq_len, n_hidden, prefix):
    """ Attention weights before applying to text, for the transformer network
        which does not aggregate attentions until the very last step. """
    h1 = TimeDistributed(Dense(n_hidden, activation='tanh'),
                         name='%s_att_TDdense1' % prefix)(input)
    h2 = TimeDistributed(Dense(1, use_bias=False),
                         name='%s_att_TDdense2' % prefix)(h1)   # [n_sample, n_steps, 1]
    flat = Reshape((seq_len,), name='%s_att_flat' % prefix)(h2) # [n_sample, n_steps]
    attention = Activation('softmax')(flat)                     # [n_sample, n_steps]    
    return attention

def lstm_attention(input, seq_len, n_hidden):
    h1 = Bidirectional(LSTM(n_hidden, dropout_W=0.5, dropout_U=0.5,
                            return_sequences=True, consume_less='gpu'))(input)
    h2 = TimeDistributed(Dense(1, use_bias=False))(h1)
    flat = Reshape((seq_len,))(h2)
    attention = Activation('softmax')(flat)
    attended_texts = Lambda(K_dot, output_shape=get_K_dot_shape)([attention, input])
    return attended_texts

def cnn_attention(input, seq_len, n_filters, filter_lengths):
    conv_embeddings = []
    for filter_length in filter_lengths:
        convolution = Conv1D(filters=n_filters, kernel_size=filter_length,
                             padding='same', activation='relu')(input)
        conv_embeddings.append(convolution)
    if len(conv_embeddings) > 1:
        combined_embedding = concatenate(conv_embeddings)
    else:
        combined_embedding = conv_embeddings[0]
    h2 = TimeDistributed(Dense(1, use_bias=False))(combined_embedding)
    flat = Reshape((seq_len,))(h2)
    attention = Activation('softmax')(flat)
    attended_texts = Lambda(K_dot, output_shape=get_K_dot_shape)([attention, input])
    return attended_texts    

def multihead_attention(input, seq_len, n_heads, n_hidden, prefix):
    # input is [n_sample, n_steps, n_hidden]
    proj_size = int(n_hidden / n_heads)
    projections = []
    for n in range(n_heads):
        projection = TimeDistributed(Dense(units=proj_size), 
                                     name='TD_proj_%s_%s' % (n, prefix))(input) # [n_sample, n_steps, proj_size]
        attention = mlp_raw_attention(input=projection, 
                                      seq_len=seq_len, 
                                      n_hidden=proj_size, 
                                      prefix='anno_wts_proj%s_%s' % (n, prefix)) # [n_sample, n_steps]
        att_proj = Lambda(K_dot, output_shape=get_K_dot_shape, 
                          name='attended_proj%s_%s' % (n, prefix))([attention, projection]) # [n_sample, proj_size]
        projections.append(att_proj)
    concat_attention = concatenate(projections, axis=-1, 
                                   name='concat_att_%s' % prefix) # [n_sample, n_hidden]
    return concat_attention

def cnn_word_encoder(max_char_seq, max_char_index, char_embed_dim=15, 
                     ngrams=range(1,8), filter_mult=25, filter_peak=None, 
                     embedding_kwargs={}, convolution_kwargs={}, n_fc_layers=1, 
                     fc_layer_type=Highway, fc_layer_kwargs={'activation': 'relu'}):
    # Char embedding layer
    char_input = Input(shape=(max_char_seq,), dtype='int32', name='char_input')
    char_embedding = Embedding(input_dim=max_char_index, output_dim=char_embed_dim,
                               input_length=max_char_seq, **embedding_kwargs, 
                               name='char_embedding')(char_input)
    # Word encoder
    ngram_embeddings = []
    for ngram in ngrams:
        if filter_peak:
            n_filters = min(filter_mult*ngram, 
                            2*filter_mult*filter_peak - filter_mult*ngram)
        else:
            n_filters = filter_mult*ngram
        filtered = Conv1D(filters=n_filters, kernel_size=ngram, 
                          **convolution_kwargs, 
                          name='char_conv_%s' % ngram)(char_embedding)
        ngram_embedding = GlobalMaxPooling1D(name='char_pool_%s' % ngram)(filtered)
        ngram_embeddings.append(ngram_embedding)
    word_embedding = concatenate(ngram_embeddings, name='char_embed_merge')
    fc_layers = []
    for n in range(n_fc_layers):
        if n == 0:
            previous_layer = word_embedding
        else:
            previous_layer = fc_layers[-1]
        fc_layer = fc_layer_type(**fc_layer_kwargs, name='word_fc%s' % n)(previous_layer)
        fc_layers.append(fc_layer)
    word_encoder = Model(inputs=char_input, outputs=fc_layers[-1], name='word_encoder')
    return word_encoder

def cnn_word_encoder2(max_char_seq, max_char_index, char_embed_dim=15, 
                      ngrams=range(1,8), filter_mult=25, filter_peak=None, 
                      convolution_kwargs={}, n_fc_layers=1,
                      fc_layer_type=Highway, 
                      fc_layer_kwargs={'activation': 'relu'}):
    char_embedding = Input(shape=(max_char_seq, char_embed_dim), 
                           name='char_embedding_input') # [n_samples, n_chars, char_embed_dim]
    # Word encoder
    ngram_embeddings = []
    for ngram in ngrams:
        if filter_peak:
            n_filters = min(filter_mult*ngram, 
                            2*filter_mult*filter_peak - filter_mult*ngram)
        else:
            n_filters = filter_mult*ngram
        filtered = Conv1D(filters=n_filters, kernel_size=ngram, 
                          **convolution_kwargs, 
                          name='char_conv_%s' % ngram)(char_embedding)
        ngram_embedding = GlobalMaxPooling1D(name='char_pool_%s' % ngram)(filtered)
        ngram_embeddings.append(ngram_embedding)
    word_embedding = concatenate(ngram_embeddings, name='char_embed_merge')
    fc_layers = []
    for n in range(n_fc_layers):
        if n == 0:
            previous_layer = word_embedding
        else:
            previous_layer = fc_layers[-1]
        fc_layer = fc_layer_type(**fc_layer_kwargs, name='word_fc%s' % n)(previous_layer)
        fc_layers.append(fc_layer)
    word_encoder = Model(inputs=char_embedding, outputs=fc_layers[-1], name='word_encoder')
    return word_encoder

def cnn_res_word_encoder(max_char_seq, max_char_index, char_embed_dim=15, 
                         ngrams=range(1,8), filter_mult=25, filter_peak=None, 
                         embedding_kwargs={}, convolution_kwargs={}, n_fc_layers=2):
    # Char embedding layer
    char_input = Input(shape=(max_char_seq,), dtype='int32', name='char_input')
    char_embedding = Embedding(input_dim=max_char_index, output_dim=char_embed_dim,
                               input_length=max_char_seq, **embedding_kwargs, 
                               name='char_embedding')(char_input)
    # Word encoder
    ngram_embeddings = []
    for ngram in ngrams:
        if filter_peak:
            n_filters = min(filter_mult*ngram, 
                            2*filter_mult*filter_peak - filter_mult*ngram)
        else:
            n_filters = filter_mult*ngram
        filtered = Conv1D(filters=n_filters, kernel_size=ngram, 
                          **convolution_kwargs, 
                          name='char_conv_%s' % ngram)(char_embedding)
        ngram_embedding = GlobalMaxPooling1D(name='char_pool_%s' % ngram)(filtered)
        ngram_embeddings.append(ngram_embedding)
    word_embedding = concatenate(ngram_embeddings, name='char_embed_merge')
    n_hidden = sum([filter_mult*ngram for ngram in ngrams])
    fc_layers = [word_embedding]
    for n in range(n_fc_layers):
        previous_layer = fc_layers[-1]
        d = Dense(units=n_hidden, activation='relu')(previous_layer)
        r = add([d, previous_layer])
        fc_layers.append(r)
    word_encoder = Model(inputs=char_input, outputs=fc_layers[-1], name='word_encoder')
    return word_encoder

def char_encoder(max_char_seq, max_char_index, char_embed_dim):
    char_input = Input(shape=(max_char_seq,), dtype='int32', name='char_input')
    char_embedding = Embedding(input_dim=max_char_index, output_dim=char_embed_dim,
                               input_length=max_char_seq, name='char_embedding')(char_input)
    return Model(inputs=char_input, outputs=char_embedding, name='char_encoder')
    
def naics_encoder_generic(input_layer, n_layers, layer_type, layer_kwargs):
    layers = [input_layer]
    for n in range(n_layers):
        previous_layer = layers[n]
        layer = layer_type(**layer_kwargs, name='naics_%s' % n)(previous_layer)
        layers.append(layer)
    return layers[-1]    

def naics_hw_only_encoder(input_layer):
    return Highway(activation='relu')(input_layer)

def naics_hw_encoder(input_layer, output_dim):
    h1 = Highway(activation='relu')(input_layer)
    return Dense(output_dim, name='naics_linear_proj')(h1)

def naics_lp_hw_encoder(input_layer, output_dim):
    l1 = Dense(output_dim, name='naics_linear_proj')(input_layer)
    h1 = Highway(activation='relu')(l1)
    return h1

def naics_hw_noproj_encoder(input_layer, n_layers=2):
    layers = [input_layer]
    for n in range(n_layers):
        previous_layer = layers[n]
        layer = Highway(activation='relu', name='naics_hw%s' % n)(previous_layer)
        layers.append(layer)
    return layers[-1]

def naics_res_encoder(input_layer, n_layers=2):
    layers = [input_layer]
    for n in range(n_layers):
        previous_layer = layers[-1]
        dense_layer = Dense(units=60, activation='relu', name='naics_dense%s' % n)(previous_layer)
        norm_layer = BatchNormalization()(dense_layer)
        residual_layer = add([norm_layer, previous_layer])
        layers.append(residual_layer)
    return layers[-1]    
    
def naics_hw2_encoder(input_layer, output_dim):
    h1 = Highway(activation='relu')(input_layer)
    h2 = Highway(activation='relu')(h1)
    return Dense(output_dim, name='naics_linear_proj')(h2)

def naics_hw3_encoder(input_layer, output_dim):
    h1 = Highway(activation='relu')(input_layer)
    h2 = Highway(activation='relu')(h1)
    h3 = Highway(activation='relu')(h2)
    return Dense(output_dim, name='naics_linear_proj')(h3)

def naics_linear_transform(input_layer, output_dim):
    return Dense(output_dim, name='naics_linear_proj')(input_layer)

def field_lstm_wide_encoder(input_shape, n_hidden, n_channels, consume_less='gpu'):
    input_layer = Input(shape=input_shape, dtype='float32')
    encoders = []
    for channel in range(n_channels):
        encoder = Bidirectional(LSTM(n_hidden, dropout_W=0.5, dropout_U=0.5,
                                     return_sequences=True, consume_less=consume_less))(input_layer)
        encoders.append(encoder)
    merged_encoder = concatenate(encoders, axis=-1)
    return Model(input=input_layer, output=merged_encoder)


class EatMask(Layer):
    """ Purpose is to remove any masking so later layers that do not
        supporting masking (like my attention layers) don't throw an error. 
        This may be better accomplished by accomodating later layers 
        (like attention layers) to handle masking, or, after upgrading, by
        using the Lambda to remove the mask, but I haven't done that yet.
    """
    def __init__(self, **kwargs):
        self.support_mask = True
        super(EatMask, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, x, mask=None):
        return x
    
    def compute_mask(self, input_shape, input_make=None):
        return None


class LayerNorm(Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """
    def __init__(self, scale_initializer='ones', bias_initializer='zeros', **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = 1e-6
        self.scale_initializer = initializers.get(scale_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],), 
                                     initializer=self.scale_initializer,
                                     trainable=True,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    trainable=True,
                                    name='{}_bias'.format(self.name))
        self.built = True
        
    def call(self, x, mask=None):
        # mean of the hidden activations
        mean = K.mean(x, axis=-1, keepdims=True) # [None, steps, hidden]
        # std of the hidden activations
        std = K.std(x, axis=-1, keepdims=True)   # [None, steps, hidden]
        norm = (x - mean) / (std + self.epsilon)
        return norm * self.scale + self.bias     # [None, steps, hidden]
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Aggregate(Layer):
    """ Aggregates outputs to capture hierarchical loss.
        aggregations - dict mapping each aggregate_label index to list of child indexes
    """
    def __init__(self, aggregations, batch_size, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.aggregations = aggregations
        self.output_size = len(aggregations)
        self.batch_size = batch_size
        
    def build(self, input_shape):
        self.built = True
        
    def call(self, x, mask=None):
        outputs = []
        for agg_index, child_indexes in sorted(self.aggregations.items()):
            subset = gather_by_columns(tensor=x, indexes=child_indexes)
            summed = K.sum(subset, axis=-1, keepdims=True)
            outputs.append(summed)
        return K.concatenate(outputs, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)
    
def gather_by_columns(tensor, indexes):
    # K.gather only lets us grab rows, so first transpose columns into rows
    t_tensor = K.transpose(tensor)        # [classes, batch_size]
    # now use gather to grab the indexed columns (now rows)
    columns = K.gather(t_tensor, indexes) # [n_indexes, batch_size]
    # now tranpose the gathered rows back to columns
    t_columns = K.transpose(columns)      # [batch_size, n_indexes]
    return t_columns
    