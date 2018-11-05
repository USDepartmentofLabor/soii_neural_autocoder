# -*- coding: utf-8 -*-
"""    
Trains an LSTM model on a single sequence of SOII narrative information
formed by concatenating all text fields and inserting tokens to indicate the
beginning of each narrative field.

Each input token is first embedded using a character-level convolutional model,
these embeddings are then fed to an LSTM to produce context sensitive embeddings.
Coding-task specific attention layers then aggregate these embeddings and the
result is concatenated with other categorical information and fed to the final
output layers to produce outputs for each coding task: 
    occupation, nature, part, event, source, and secondary source

"""
import math
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Input, BatchNormalization
from keras.layers.merge import concatenate, add
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

from cac_net.utils import get_train_test, valid_codes, Classifier, jdump
from cac_net.custom_layers import dynamic_attention, naics_res_encoder, cnn_word_encoder
from cac_net.metrics import macro_f1
from cac_net.preprocessing import CharTokenizer, NAICSVectorizer, Labeler
from cac_net.preprocessing import LambdaVectorizer, extract_job_category, MetaVectorizer
from cac_net.batch_preprocessing import WordCharVectorizer, generate_meta, FieldVectorizer
from cac_net.batch_preprocessing import pretokenize_rows
from cac_net.masking import mask_from_embedded_seq, masked_seq
import cac_net.constants

max_sequence_length = 180
max_word_seq = None
max_char_seq = 15
batch_size = 64
char_embed_dim = 15
field_hidden = 1024
filter_mult = 25

occ_text_fields = ['occupation_text', 'other_text', 'company_name', 
                   'secondary_name', 'unit_description']
nar_text_fields = ['nar_activity', 'nar_event', 'nar_nature', 'nar_source']
text_fields = occ_text_fields + nar_text_fields
labels = ['soc', 'nature_code', 'part_code', 'event_code', 'source_code', 'sec_source_code']

print('Loading data...')
data = {}
data_file = os.path.join(cac_net.constants.DATA_DIR, 'training.csv')
data['train'], data['test'] = get_train_test(train_years=(2014,2014),
                                             test_years=(2015,2015),
                                             data_source=data_file,
                                             create_cache=False)
data['train'] = pretokenize_rows(data['train'], fields=text_fields, 
                                 max_length=max_sequence_length,
                                 field_start_token='_',
                                 field_end_token=None,
                                 field_suffix_fn=lambda n, f: '%s_' % n)
data['test'] = pretokenize_rows(data['test'], fields=text_fields,
                                max_length=max_sequence_length,
                                field_start_token='_',
                                field_end_token=None,
                                field_suffix_fn=lambda n, f: '%s_' % n)
n_train = len(data['train'])
n_test = len(data['test'])

vect_ins_outs = []

# Character tokenizer
all_tokens = [row['all_tokens'] for row in data['train'] + data['test']]
char_tokenizer = CharTokenizer(add_borders=True)
wc_vectorizer = WordCharVectorizer(max_word_seq=max_word_seq, 
                                   max_char_seq=max_char_seq, 
                                   char_tokenizer=char_tokenizer,
                                   n_char_field_indicators=len(text_fields))
wc_vectorizer.fit(all_tokens)
n_char_features = len(wc_vectorizer.char_tokenizer.character_map)
vect_ins_outs.append((wc_vectorizer, 'all_tokens', 'all_tokens'))

# Field vectorizer
field_vectorizer = FieldVectorizer(fields=text_fields)
vect_ins_outs.append((field_vectorizer, None, 'field_indicators'))

# Create NAICS vectorizer
train_naics = [row['naics'] for row in data['train']]
test_naics = [row['naics'] for row in data['test']]
naics_vectorizer = NAICSVectorizer()
naics_vectorizer.fit(train_naics + test_naics)
vect_ins_outs.append((naics_vectorizer, 'naics', 'naics'))

# Job Category vectorizer 
jc_tokens = set(['cat=%s' % cat for cat in cac_net.constants.JOB_CATEGORIES])
jc_vectorizer = LambdaVectorizer([extract_job_category], tokens=jc_tokens)
vect_ins_outs.append((jc_vectorizer, None, 'job_category'))

# Label vectorizers
n_classes = {}
for label in labels:
    labeler = Labeler()
    labeler.fit(valid_codes(label))
    n_classes[label] = len(valid_codes(label))
    vect_ins_outs.append((labeler, label, label))

# MetaVectorizer - a vectorizer which combines the operations of all other vectorizers
meta_vectorizer = MetaVectorizer(vect_ins_outs)

generator = {}
for dataset in ['train', 'test']:
    input_fields = ['all_tokens', 'field_indicators'] + ['naics', 'job_category']
    generator[dataset] = generate_meta(rows=data[dataset], 
                                       meta_vectorizer=meta_vectorizer,
                                       input_fields=input_fields,
                                       output_fields=labels, 
                                       batch_size=batch_size,
                                       max_sequence_length=max_sequence_length)

print('Define model')
# Word encoder
word_encoder = cnn_word_encoder(max_char_seq=max_char_seq, 
                                max_char_index=n_char_features+1,
                                char_embed_dim=15, ngrams=range(1,8),
                                filter_mult=filter_mult, 
                                convolution_kwargs={'activation':'relu'})

# Text encoder
text_encoder = Bidirectional(LSTM(field_hidden, 
                                  dropout=0.5, 
                                  recurrent_dropout=0.5, 
                                  return_sequences=True,
                                  name='text_BiLSTM'))

# Text inputs
text_input = Input(shape=(None, max_char_seq), dtype='float32', 
                   name='all_tokens') # [n_samp, n_word_seq, n_char_seq]
encoded_words = TimeDistributed(word_encoder, 
                                name='encoded_joint_text')(text_input) # [n_samp, n_word_seq, n_hidden]

# Field indicator embedding
field_input = Input(shape=(None, len(text_fields)), dtype='float32', 
                    name='field_indicators')
encoded_fields = TimeDistributed(Dense(encoded_words.shape[-1].value), 
                                 name='field_indicator_embedding')(field_input)

encoded_word_fields = add([encoded_words, encoded_fields])

# mask the blank inputs to the LSTM
mask = mask_from_embedded_seq(text_input)
masked_combined_words = masked_seq(encoded_word_fields, mask)
field_embedding = text_encoder(masked_combined_words) # [n_samp, n_seq, n_hidden]
# mask the LSTM outputs corresponding to the blank inputs
masked_field_embedding = masked_seq(field_embedding, mask)

# NAICS input
naics_input = Input(shape=(60,), dtype='float32', name='naics')
naics_embedding = naics_res_encoder(input_layer=naics_input, n_layers=3)

# Job category
job_category_input = Input(shape=(12,), dtype='float32', name='job_category')

# All outputs
outputs = []
for label in labels:
    # merged_texts is [n_sample, n_texts, n_features]
    aux_layers = [naics_embedding]
    if label == 'soc':
        aux_layers.append(job_category_input)
    hidden_units = masked_field_embedding.shape[-1].value
    dropped_texts = Dropout(0.5, noise_shape=(batch_size, 1, hidden_units))(masked_field_embedding)
    attended_texts = dynamic_attention(input_layer=dropped_texts, 
                                       prefix=label,
                                       mask=mask)
    merged_inputs = concatenate([attended_texts] + aux_layers,
                                name='all_features_%s' % label)
    merged_hidden = merged_inputs.shape[-1].value
    dropout_1 = Dropout(0.5, name='%s_hw1_dropout' % label)(merged_inputs)
    d1 = Dense(units=merged_hidden, activation='relu', name='%s_hw1' % label)(dropout_1)
    b1 = BatchNormalization()(d1)
    r1 = add([b1, dropout_1])
    dropout_2 = Dropout(0.5, name='%s_hw2_dropout' % label)(r1)
    d2 = Dense(units=merged_hidden, activation='relu', name='%s_hw2' % label)(dropout_2)
    b2 = BatchNormalization()(d2)
    r2 = add([b2, dropout_2])
    dropout_3 = Dropout(0.5, name='%s_softmax_dropout' % label)(r2)
    output = Dense(n_classes[label], activation='softmax', name=label)(dropout_3)
    outputs.append(output)

model = Model(inputs=[text_input, field_input, naics_input, job_category_input], 
              outputs=outputs)
optimizer = Adam(clipnorm=1., lr=.0004)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
              metrics=['accuracy', macro_f1])

print('Train...')
csv_logger = CSVLogger('log.csv')
check_pointer = ModelCheckpoint('big_single_seq_180_{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True)
early_stopper = EarlyStopping(monitor='val_loss', patience=5)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=1, 
                               factor=0.5, verbose=1)
clf = Classifier(meta_vectorizer=meta_vectorizer, model_path='')
jdump(clf, '')
model.fit_generator(generator=generator['train'], 
                    steps_per_epoch=math.floor(n_train/batch_size),
                    validation_data=generator['test'], 
                    validation_steps=math.floor(n_test/batch_size),
                    epochs=100, 
                    callbacks=[check_pointer, csv_logger, early_stopper, lr_reducer])
