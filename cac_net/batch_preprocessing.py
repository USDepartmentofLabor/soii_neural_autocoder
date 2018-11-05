# -*- coding: utf-8 -*-
"""
Batch preprocessing automatically assembles same sequence length batches for
ultra-fast processing. Requires pre-tokenized inputs for efficient bucketing
and field embedding.

@author: MEASURE_A
"""
import random
import numpy as np
import cac_net.preprocessing


# template for the field_tokens and field_tokens_len fields
TOKENS_FIELD = '%s_tokens'
TOKENS_FIELD_LEN = '%s_len'
TOKENS_ALL = 'all_tokens'
TOKENS_TOTAL_LEN = 'tokens_total_len'
TOKENS_FUZZ_LEN = 'tokens_fuzz_len'


def texts_to_word_char_seq(tokenized_texts, char_tokenizer, max_word_seq, 
                           max_char_seq, char_field_indicator=None):
    """ A mask and dynamic max_word_seq friendly version of texts_to_word_char_seq """
    n_texts = len(tokenized_texts)
    # infer max_word_seq length if None provided
    if not max_word_seq:
        max_word_seq = max([len(words) for words in tokenized_texts])
    output = np.zeros(shape=(n_texts, max_word_seq, max_char_seq), dtype=np.int)
    for text_idx, words in enumerate(tokenized_texts):
        for word_idx, word in enumerate(words[0: max_word_seq]):
            if len(word) > 0:
                # Only pad if there is a word to pad
                if char_field_indicator is not None:
                    word = char_field_indicator + word
            chars = char_tokenizer.texts_to_sequences([word])[0]
            for char_idx, char in enumerate(chars[0: max_char_seq]):
                output[text_idx, word_idx, char_idx] = char
    return output

def generate_meta(rows, meta_vectorizer, input_fields, output_fields, 
                  batch_size, max_sequence_length, 
                  sequence_length_field=TOKENS_TOTAL_LEN):
    """ Adds dynamic bucketing by sequence size to speed processing. Requires
        that we sort by text field sequence size first, which means we need to compute
        it first. 
        
        input_fields: vectorized fields that will be inputs to the model
        output_fields: vectorized fields that will be target outputs of the model
    """
    np.random.seed(10)
    buckets = get_fuzzy_shuffled_buckets(rows=rows, batch_size=batch_size,
                                         max_sequence_length=max_sequence_length,
                                         sequence_length_field=sequence_length_field,
                                         fuzz_int=1, 
                                         biggest_first=True)
    n_buckets = len(buckets)
    start = 0
    while 1:
        if start > n_buckets - 1:
            # reshuffle and reset our start and end points
            buckets = get_fuzzy_shuffled_buckets(rows=rows, batch_size=batch_size,
                                                 max_sequence_length=max_sequence_length,
                                                 sequence_length_field=sequence_length_field,
                                                 fuzz_int=5, 
                                                 biggest_first=False)
            start = 0        
        # Generate outputs
        output = meta_vectorizer.transform(buckets[start])
        X_out = {i: output[i] for i in input_fields}
        Y_out = {i: output[i] for i in output_fields}
        # Increment start
        start += 1
        yield (X_out, Y_out)

def fuzz_sequence_length(rows, sequence_length_field=TOKENS_TOTAL_LEN, fuzz_int=5):
    fuzz_rows = []
    for row in rows:
        fuzz_row = row.copy()
        fuzz_row[TOKENS_FUZZ_LEN] = fuzz_row[sequence_length_field] + np.random.randint(fuzz_int)
        fuzz_rows.append(fuzz_row)
    return fuzz_rows

def get_buckets(rows, batch_size, max_sequence_length, sequence_length_field=TOKENS_TOTAL_LEN):
    print('generating buckets')
    rows = sorted(rows, key=lambda x: x[TOKENS_FUZZ_LEN], reverse=True)
    f_rows = [row for row in rows if row[sequence_length_field] <= max_sequence_length]
    n_dropped = len(rows) - len(f_rows)
    print('%s inputs dropped for exceeding max length of %s' % (n_dropped, max_sequence_length))
    n_buckets = len(f_rows) // batch_size
    buckets = []
    for bucket_n in range(n_buckets):
        bucket = f_rows[bucket_n * batch_size: (bucket_n + 1) * batch_size]
        buckets.append(bucket)
    return buckets

def get_fuzzy_shuffled_buckets(rows, batch_size, max_sequence_length, 
                               sequence_length_field=TOKENS_TOTAL_LEN, 
                               fuzz_int=5, biggest_first=False):
    fuzz_rows = fuzz_sequence_length(rows=rows, 
                                     sequence_length_field=sequence_length_field,
                                     fuzz_int=fuzz_int)
    buckets = get_buckets(rows=fuzz_rows, 
                          batch_size=batch_size, 
                          max_sequence_length=max_sequence_length,
                          sequence_length_field=sequence_length_field)
    if biggest_first:
        # buckets is always sorted biggest to smallest so memory fails happen early
        biggest_bucket = buckets[0]
        other_buckets = buckets[1:]
        random.shuffle(other_buckets)
        buckets = [biggest_bucket] + other_buckets
    else:
        random.shuffle(buckets)
    return buckets


class WordCharVectorizer(cac_net.preprocessing.WordCharVectorizerEOS):
    def fit(self, tokenized_texts):
        words = []
        words.extend(self.word_field_indicators)
        words.extend(self.char_field_indicators)
        for tokenized_text in tokenized_texts:
            words.extend(tokenized_text)
        self.char_tokenizer.fit_on_texts(words)
    
    def transform(self, texts, field=None, field_indicator=None):
        # Get the field indicator, if necessary
        if len(self.char_field_indicators) > 0:
            char_field_indicator = self.char_field_indicators[field_indicator]
        else:
            char_field_indicator = None
        # Get the appropriate word sequence, if field specific ones were assigned
        if hasattr(self.max_word_seq, 'get'):
            max_word_seq = self.max_word_seq[field]
        else:
            max_word_seq = self.max_word_seq
        if hasattr(self.max_char_seq, 'get'):
            max_char_seq = self.max_char_seq[field]
        else:
            max_char_seq = self.max_char_seq
        return texts_to_word_char_seq(tokenized_texts=texts,
                                      char_tokenizer=self.char_tokenizer,
                                      max_word_seq=max_word_seq, 
                                      max_char_seq=max_char_seq,
                                      char_field_indicator=char_field_indicator)   

class FieldVectorizer:
    """ ins should be None to indicate inputs are the rows """
    def __init__(self, fields, n_tokens_field=TOKENS_TOTAL_LEN):
        self.fields = fields
        self.n_fields = len(self.fields)
        self.n_tokens_field = n_tokens_field
    
    def fit(self, rows):
        pass
    
    def transform(self, rows):
        n_rows = len(rows)
        n_tokens = max([row[self.n_tokens_field] for row in rows])
        output = np.zeros((n_rows, n_tokens, self.n_fields))
        for row_id, row in enumerate(rows):
            token_pos = 0
            for field_id, field in enumerate(self.fields):
                for i in range(row[TOKENS_FIELD_LEN % field]):
                    output[row_id, token_pos, field_id] = 1
                    token_pos += 1
        return output

    def inverse_transform(self, vector):
        print('inverse transform not implemented for FieldVectorizer')

def pretokenize_rows(rows, fields, max_length=200, 
                     field_start_token='<S>', 
                     field_end_token='</S>',
                     case_start_token=None,
                     case_end_token=None,
                     field_suffix_fn=lambda n, field: ''):
    """ Adds %s_tokens, %s_len, and %s_len for each field and row. """
    word_tokenizer = cac_net.preprocessing.WordTokenizer()
    for row in rows:
        tokens_total_len = 0
        row[TOKENS_ALL] = []
        for n, field in enumerate(fields):
            tokens = word_tokenizer.tokenize(row[field])
            if field_start_token:
                tokens.insert(0, field_start_token + field_suffix_fn(n, field))
            if field_end_token:
                tokens.append(field_end_token + field_suffix_fn(n, field))
            if case_start_token and n==0:
                tokens.insert(0, case_start_token)
            if case_end_token and n==(len(fields)-1):
                tokens.append(case_end_token)
            tokens_len = len(tokens)
            tokens_total_len += len(tokens)
            row[TOKENS_FIELD % field] = tokens
            row[TOKENS_FIELD_LEN % field] = tokens_len
            row[TOKENS_ALL] += tokens
        row[TOKENS_TOTAL_LEN] = tokens_total_len
    n_orig = len(rows)
    rows = [row for row in rows if row[TOKENS_TOTAL_LEN] <= max_length]
    n_after = len(rows)
    print('%s rows dropped for exceeding max token length' % (n_orig - n_after))
    return rows


class PreTokenizer:
    def __init__(self, fields, max_length=200, 
                 field_start_format="<{}>", 
                 field_end_format="</{}>"):
        tokenizer = cac_net.preprocessing.WordTokenizer()
        self.tokenizer = tokenizer
        self.fields = fields
        self.max_length = max_length
        self.field_start_format = field_start_format
        self.field_end_format = field_end_format
    
    def transform(self, rows):
        for row in rows:
            tokens_total_len = 0
            row[TOKENS_ALL] = []
            for n, field in enumerate(self.fields):
                tokens = self.tokenizer.tokenize(row[field])
                if self.field_start_format:
                    field_start_token = self.field_start_format.format(n)
                    tokens.insert(0, field_start_token)
                if self.field_end_format:
                    field_end_token = self.field_end_format.format(n)
                    tokens.append(field_end_token)
                tokens_len = len(tokens)
                tokens_total_len += len(tokens)
                row[TOKENS_FIELD % field] = tokens
                row[TOKENS_FIELD_LEN % field] = tokens_len
                row[TOKENS_ALL] += tokens
            row[TOKENS_TOTAL_LEN] = tokens_total_len
        n_orig = len(rows)
        rows = [row for row in rows if row[TOKENS_TOTAL_LEN] <= self.max_length]
        n_after = len(rows)
        print('%s rows dropped for exceeding max token length' % (n_orig - n_after))
        return rows
        