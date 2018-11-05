# -*- coding: utf-8 -*-
"""
Text encoding utilities for CAC_net
"""
import re
import inspect
import warnings
import collections
import numpy as np
np.random.seed(1337)
import nltk
import cac_net.constants
import cac_net.sampling


def texts_to_word_char_seq(texts, word_tokenizer, char_tokenizer, max_word_seq, 
                           max_char_seq, word_field_indicator=None, 
                           char_field_indicator=None, eos_indicator=None):
    n_texts = len(texts)
    output = np.zeros(shape=(n_texts, max_word_seq, max_char_seq), dtype=np.int)
    for text_idx, text in enumerate(texts):
        words = word_tokenizer.tokenize(text)
        if word_field_indicator is not None:
            words.insert(0, word_field_indicator)
        if eos_indicator:
            words.append(eos_indicator)
        for word_idx, word in enumerate(words[0: max_word_seq]):
            if char_field_indicator is not None:
                word = char_field_indicator + word
            chars = char_tokenizer.texts_to_sequences([word])[0]
            for char_idx, char in enumerate(chars[0: max_char_seq]):
                output[text_idx, word_idx, char_idx] = char
    return output

def texts_to_word_char_seq_mask(texts, word_tokenizer, char_tokenizer, max_word_seq, 
                                max_char_seq, word_field_indicator=None, 
                                char_field_indicator=None, eos_indicator=None):
    """ A mask friendly version of texts_to_word_char_seq """
    n_texts = len(texts)
    output = np.zeros(shape=(n_texts, max_word_seq, max_char_seq), dtype=np.int)
    for text_idx, text in enumerate(texts):
        words = word_tokenizer.tokenize(text)
        if len(words) > 0:
            # Only add special indicator if there is a text input for this field
            if word_field_indicator is not None:
                words.insert(0, word_field_indicator)
            if eos_indicator:
                words = words[0: max_word_seq - 1] # make sure there's always room for EOS indicator
                words.append(eos_indicator)
        for word_idx, word in enumerate(words[0: max_word_seq]):
            if len(word) > 0:
                # Only pad if there is a word to pad
                if char_field_indicator is not None:
                    word = char_field_indicator + word
            chars = char_tokenizer.texts_to_sequences([word])[0]
            for char_idx, char in enumerate(chars[0: max_char_seq]):
                output[text_idx, word_idx, char_idx] = char
    return output


def texts_to_sent_word_char_seq(texts, char_tokenizer, max_sent_seq,
                                max_word_seq, max_char_seq):
    n_texts = len(texts)
    output_shape = (n_texts, max_sent_seq, max_word_seq, max_char_seq)
    output = np.zeros(shape=output_shape, dtype=np.int)
    for text_idx, text in enumerate(texts):
        sentences = nltk.sent_tokenize(text)
        for sent_idx, sentence in enumerate(sentences[0: max_sent_seq]):
            words = nltk.word_tokenize(text)
            for word_idx, word in enumerate(words[0: max_word_seq]):
                chars = char_tokenizer.texts_to_sequences([word])[0]
                for char_idx, char in enumerate(chars[0: max_char_seq]):
                    output[text_idx, word_idx, char_idx] = char
    return output
           

def generate_meta(rows, meta_vectorizer, input_fields, output_fields, batch_size):
    """ input_fields: vectorized fields that will be inputs to the model
        output_fields: vectorized fields that will be target outputs of the model
    """
    np.random.seed(10)
    if not hasattr(rows, 'shape'):
        rows = np.asarray(rows)
    n_samples = len(rows)
    start = 0
    while 1:
        stop = start + batch_size
        if stop > n_samples:
            # reshuffle and reset our start and end points
            random_index = np.random.permutation(n_samples)
            rows = rows[random_index]      
            start = 0
            stop = start + batch_size         
        # Generate outputs
        output = meta_vectorizer.transform(rows[start: stop])
        X_out = {i: output[i] for i in input_fields}
        Y_out = {i: output[i] for i in output_fields}
        # Increment start
        start = stop
        yield (X_out, Y_out)

            
def generate_even_sample(rows, meta_vectorizer, input_fields, output_fields, 
                         batch_size, sampled_field):
    """ input_fields: vectorized fields that will be inputs to the model
        output_fields: vectorized fields that will be target outputs of the model
    """
    np.random.seed(10)    
    n_samples = len(rows)
    indexed_rows = cac_net.sampling.get_indexed_rows(rows, sampled_field)
    n_codes = len(indexed_rows)
    sampled_rows = cac_net.sampling.even_sample(indexed_rows, sampled_field, 
                                                n_codes, n_samples)
    start = 0
    while 1:
        stop = start + batch_size
        if stop > n_samples:
            # reshuffle and reset our start and end points
            sampled_rows = cac_net.sampling.even_sample(indexed_rows, sampled_field,
                                                        n_codes, n_samples)
            start = 0
            stop = start + batch_size         
        # Generate outputs
        output = meta_vectorizer.transform(sampled_rows[start: stop])
        X_out = {i: output[i] for i in input_fields}
        Y_out = {i: output[i] for i in output_fields}
        # Increment start
        start = stop
        yield (X_out, Y_out)

def generate_alternating_even_sample(rows, meta_vectorizer, input_fields, 
                                     output_fields, batch_size, sampled_fields):
    """ input_fields: vectorized fields that will be inputs to the model
        output_fields: vectorized fields that will be target outputs of the model
    """
    np.random.seed(10)    
    n_samples = len(rows)
    alternating_fields = list(np.random.permutation(sampled_fields))
    sampled_field = alternating_fields.pop()
    indexed_rows = cac_net.sampling.get_indexed_rows(rows, sampled_field)
    n_codes = len(indexed_rows)
    sampled_rows = cac_net.sampling.even_sample(indexed_rows, sampled_field, 
                                                n_codes, n_samples)
    start = 0
    while 1:
        stop = start + batch_size
        if stop > n_samples:
            # reshuffle and reset our start and end points
            if len(alternating_fields) == 0:
                alternating_fields = list(np.random.permutation(sampled_fields))
            sampled_field = alternating_fields.pop()
            sampled_rows = cac_net.sampling.even_sample(indexed_rows, sampled_field,
                                                        n_codes, n_samples)
            start = 0
            stop = start + batch_size         
        # Generate outputs
        output = meta_vectorizer.transform(sampled_rows[start: stop])
        X_out = {i: output[i] for i in input_fields}
        Y_out = {i: output[i] for i in output_fields}
        # Increment start
        start = stop
        yield (X_out, Y_out)


class MetaVectorizer:
    def __init__(self, vectorizer_inputs_outputs):
        """ vectorizer_inputs_outputs should be a 3 tuple of an already 
            trained vectorizer with the input fields it will operate on and
            the corresponding output fields it will generate. 
            Each vectorizer must have a transform and inverse_transform method.
        """
        # verify that vectorizer_fields is a dictionary
        self.vect_ins_outs = []
        for vect, ins, outs in vectorizer_inputs_outputs:
            # Force ins and outs to lists
            if isinstance(ins, (str, type(None))):
                ins = [ins]
            if isinstance(outs, (str, type(None))):
                outs = [outs]
            self.vect_ins_outs.append((vect, ins, outs))
           
    def transform(self, rows):
        """ Return a {field: vector} dictionary of the transformed data """
        output = {}
        for vectorizer, ins, outs in self.vect_ins_outs:
            for n, input_field in enumerate(ins):
                output_field = outs[n]
                # Generate the transform inputs, either field specific or generic
                if input_field:
                    inputs = [row[input_field] for row in rows]
                else:
                    inputs = rows
                # Set the field indicator if appropriate
                args = [inputs]
                kwargs = {}
                ### Check whether vectorizer expects additional information and send it
                if 'field_indicator' in inspect.getargspec(vectorizer.transform).args:
                    kwargs['field_indicator'] = n
                # Set the field value, if appropriate
                if 'field' in inspect.getargspec(vectorizer.transform).args:
                    kwargs['field'] = input_field
                # Attempt vectorization
                try:
                    transformed = vectorizer.transform(*args, **kwargs)
                except AttributeError:
                    msg = 'no transform method found for %s, using texts_to_sequences instead' % str(vectorizer)
                    warnings.warn(msg)
                    transformed = vectorizer.texts_to_sequences(*args, **kwargs)
                # Verify vectorization
                assert len(inputs) == len(transformed), '# of inputs and outputs should be equal'
                output[output_field] = transformed
        return output

    def inverse_transform(self, field_vectors):
        """ Field vectors should be a {field: vector} dictionary representing
            one or more training examples. Only provides correct output when
            each field is associated with only a single vectorizer.
        """
        output = {}
        for field, vectors in field_vectors.items():
            vectorizers = [v for v, i, o in self.vect_ins_outs if field in o]
            if len(vectorizers) != 1:
                msg = '%s vectorizers are associated with field %s, results unstable' % (len(vectorizers), field)
                raise ValueError(msg)
            vectorizer = vectorizers[0]
            output[field] = vectorizer.inverse_transform(vectors)
        return output
    
    def get_vectorizer_by_input(self, input_field):
        """ Given an input_field, return the vectorizer that transforms it. """
        vectorizers = [v for v, i, o in self.vect_ins_outs if input_field in i]
        assert len(vectorizers) == 1, 'expected 1 vectorizer, %s retrieved' % len(vectorizers)
        return vectorizers[0]


# JUST for testing purposes
class WordCharVectorizer:
    def __init__(self, max_word_seq, max_char_seq, char_tokenizer=None,
                 n_word_field_indicators=None, n_char_field_indicators=None):
        """ 
        max_word_seq - number or dictionary mapping fields to numbers
        max_char_seq - number or dictionary mapping fields to numbers
        n_word_field_indicators - the number of fields that should be 
            distinguished by a special word token at the start of each field.
        n_char_field_indicators - the number of fields that should be 
            distinguished by a special char token at the start of each word.
        """
        if not char_tokenizer:
            print('no char_tokenizer provided, initializing default')
            char_tokenizer = CharTokenizer()
        self.word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.char_tokenizer = char_tokenizer
        self.max_word_seq = max_word_seq
        self.max_char_seq = max_char_seq
        self.word_field_indicators = []
        self.char_field_indicators = []
        if n_word_field_indicators:
            for i in range(n_word_field_indicators):
                self.word_field_indicators.append(chr(40000 + i))
        if n_char_field_indicators:
            for i in range(n_char_field_indicators):
                self.char_field_indicators.append(chr(50000 + i))
        
    def fit(self, texts):
        words = []
        words.extend(self.word_field_indicators)
        words.extend(self.char_field_indicators)
        for text in texts:
            words.extend(self.word_tokenizer.tokenize(text))
        self.char_tokenizer.fit_on_texts(words)
                
    def transform(self, texts, field=None, field_indicator=None):
        # Get the field indicator, if necessary
        if len(self.word_field_indicators) > 0:
            word_field_indicator = self.word_field_indicators[field_indicator]
        else:
            word_field_indicator = None
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
        return texts_to_word_char_seq(texts=texts,
                                      word_tokenizer=self.word_tokenizer,
                                      char_tokenizer=self.char_tokenizer,
                                      max_word_seq=max_word_seq, 
                                      max_char_seq=max_char_seq,
                                      word_field_indicator=word_field_indicator,
                                      char_field_indicator=char_field_indicator)
    
    def inverse_transform(self, vector):
        assert len(vector.shape) == 3, 'unexpected vector shape'
        output = []
        for row in vector:
            words = []
            for word in row:
                chars = []
                for char_index in word:
                    if char_index == 0:
                        continue
                    char = self.char_tokenizer.reverse_map[char_index]
                    if char not in ['start', 'end']:
                        chars.append(char)
                if chars:
                    words.append(''.join(chars))
            output.append(' '.join(words))
        return output    

class WordCharVectorizerEOS:
    def __init__(self, max_word_seq, max_char_seq, char_tokenizer=None, 
                 word_tokenizer=None, n_word_field_indicators=None, 
                 n_char_field_indicators=None, eos_indicator=False):
        """ 
        max_word_seq - number or dictionary mapping fields to numbers
        max_char_seq - number or dictionary mapping fields to numbers
        n_word_field_indicators - the number of fields that should be 
            distinguished by a special word token at the start of each field.
        n_char_field_indicators - the number of fields that should be 
            distinguished by a special char token at the start of each word.
        """
        if not char_tokenizer:
            print('no char_tokenizer provided, initializing default')
            char_tokenizer = CharTokenizer()
        if not word_tokenizer:
            print('no word_tokenizer provided, initializing WordTokenizer')
            word_tokenizer = WordTokenizer()
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        self.max_word_seq = max_word_seq
        self.max_char_seq = max_char_seq
        self.word_field_indicators = []
        self.char_field_indicators = []
        if n_word_field_indicators:
            for i in range(n_word_field_indicators):
                self.word_field_indicators.append(chr(40000 + i))
        if n_char_field_indicators:
            for i in range(n_char_field_indicators):
                self.char_field_indicators.append(chr(50000 + i))
        if eos_indicator:
            self.eos_indicator = chr(70000)
        else:
            self.eos_indicator = None
        
    def fit(self, texts):
        words = []
        if self.eos_indicator:
            words.append(self.eos_indicator)
        words.extend(self.word_field_indicators)
        words.extend(self.char_field_indicators)
        for text in texts:
            words.extend(self.word_tokenizer.tokenize(text))
        self.char_tokenizer.fit_on_texts(words)
                
    def transform(self, texts, field=None, field_indicator=None):
        # Get the field indicator, if necessary
        if len(self.word_field_indicators) > 0:
            word_field_indicator = self.word_field_indicators[field_indicator]
        else:
            word_field_indicator = None
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
        return texts_to_word_char_seq(texts=texts,
                                      word_tokenizer=self.word_tokenizer,
                                      char_tokenizer=self.char_tokenizer,
                                      max_word_seq=max_word_seq, 
                                      max_char_seq=max_char_seq,
                                      word_field_indicator=word_field_indicator,
                                      char_field_indicator=char_field_indicator,
                                      eos_indicator=self.eos_indicator)
    
    def inverse_transform(self, vector):
        assert len(vector.shape) == 3, 'unexpected vector shape'
        output = []
        for row in vector:
            words = []
            for word in row:
                chars = []
                for char_index in word:
                    if char_index == 0:
                        continue
                    char = self.char_tokenizer.reverse_map[char_index]
                    if char not in ['start', 'end']:
                        chars.append(char)
                if chars:
                    words.append(''.join(chars))
            output.append(' '.join(words))
        return output
    
    
class WordCharVectorizerMask(WordCharVectorizerEOS):
    def transform(self, texts, field=None, field_indicator=None):
        # Get the field indicator, if necessary
        if len(self.word_field_indicators) > 0:
            word_field_indicator = self.word_field_indicators[field_indicator]
        else:
            word_field_indicator = None
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
        return texts_to_word_char_seq_mask(texts=texts,
                                           word_tokenizer=self.word_tokenizer,
                                           char_tokenizer=self.char_tokenizer,
                                           max_word_seq=max_word_seq, 
                                           max_char_seq=max_char_seq,
                                           word_field_indicator=word_field_indicator,
                                           char_field_indicator=char_field_indicator,
                                           eos_indicator=self.eos_indicator)    

        
class Labeler:
    def __init__(self):
        pass
    
    def fit(self, labels):
        labels = set(labels)
        self.label_map = {}
        for n, label in enumerate(labels):
            self.label_map[label] = n
        self.reverse_index = {n: label for label, n in self.label_map.items()}
    
    def transform(self, labels):
        transformed = []
        for label in labels:
            transformed.append(self.label_map[label])
        transformed = np.array(transformed)
        one_hot = np.zeros(shape=(len(labels), len(self.label_map)))
        one_hot[np.arange(len(labels)), transformed] = 1
        return one_hot

    def inverse_transform(self, vector):
        assert len(vector.shape) == 2, 'unexpected vector shape'
        output = []
        for row in vector:
            index = row.argmax()
            output.append(self.reverse_index[index])
        return output


class DetailLabeler:
    def __init__(self, detail=None):
        self.detail = detail
    
    def filter_label(self, label):
        label = re.sub('[ -]', '', label)
        return label[0: self.detail]
    
    def filter_labels(self, labels):
        return [self.filter_label(label) for label in labels]
    
    def fit(self, labels):
        labels = set(labels)
        labels = set(self.filter_labels(labels))
        self.n_labels = len(labels)
        self.label_map = {}
        for n, label in enumerate(labels):
            self.label_map[label] = n
        self.reverse_index = {n: label for label, n in self.label_map.items()}
    
    def transform(self, labels):
        labels = self.filter_labels(labels)
        transformed = []
        for label in labels:
            transformed.append(self.label_map[label])
        transformed = np.array(transformed)
        one_hot = np.zeros(shape=(len(labels), len(self.label_map)))
        one_hot[np.arange(len(labels)), transformed] = 1
        return one_hot

    def inverse_transform(self, vector):
        assert len(vector.shape) == 2, 'unexpected vector shape'
        output = []
        for row in vector:
            index = row.argmax()
            output.append(self.reverse_index[index])
        return output
        

class FlexLabeler:
    """ Has a special unseen label for labels not seen in training. """
    def __init__(self, unseen='unseen'):
        self.unseen = unseen
    
    def fit(self, labels):
        labels = set(labels)
        labels.add(self.unseen)
        self.label_map = {}
        for n, label in enumerate(labels):
            self.label_map[label] = n
        self.reverse_index = {n: label for label, n in self.label_map.items()}
        self.labels = labels
    
    def transform(self, labels):
        transformed = []
        for label in labels:
            if label in self.labels:    
                transformed.append(self.label_map[label])
            else:
                transformed.append(self.label_map[self.unseen])
        transformed = np.array(transformed)
        one_hot = np.zeros(shape=(len(labels), len(self.label_map)))
        one_hot[np.arange(len(labels)), transformed] = 1
        return one_hot

    def inverse_transform(self, vector):
        assert len(vector.shape) == 2, 'unexpected vector shape'
        output = []
        for row in vector:
            index = row.argmax()
            output.append(self.reverse_index[index])
        return output


class CharTokenizer:
    def __init__(self, add_borders=False, lowercase=False):
        self.add_borders = add_borders
        self.lowercase = lowercase
    
    def fit_on_texts(self, texts):
        self.characters = set()
        self.characters.add('unknown')
        if self.add_borders:
            self.characters.add('start')
            self.characters.add('end')
        for text in texts:
            self._verify_text(text)
            if self.lowercase:
                text = text.lower()
            for char in text:
                self.characters.add(char)
        self.character_map = collections.OrderedDict()
        self.reverse_map = {}
        for n, character in enumerate(sorted(self.characters)):
            index = n + 1 # reserve 0 for blanks
            self.character_map[character] = index
            self.reverse_map[index] = character
        self.max_index = max(self.reverse_map)
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            self._verify_text(text)
            sequence = []
            if self.lowercase:
                text = text.lower()
            for character in text:
                try:
                    index = self.character_map[character]
                except KeyError:
                    index = self.character_map['unknown']                   
                sequence.append(index)
            if self.add_borders:
                sequence.insert(0, self.character_map['start'])
                sequence.append(self.character_map['end'])
            sequences.append(sequence)
        return sequences

    def _verify_text(self, text):
        if not isinstance(text, str):
            msg = 'expected text to be string, found %s instead' % type(text)
            raise ValueError(msg)
        

class NAICSVectorizer:
    def __init__(self):
        pass        

    def fit(self, naics_codes):
        for naics_code in naics_codes:
            assert len(naics_code) == 6, 'naics code must be exactly 6 digits long, found: %d' % len(naics_code)
            assert naics_code.isdigit(), 'naics code must be a number, found: %s' % naics_code
        
    def transform(self, naics_codes):
        outputs = []
        for naics_code in naics_codes:
            assert len(naics_code) == 6, 'naics code must be exactly 6 digits long, found: %d' % len(naics_code)
            assert naics_code.isdigit(), 'naics code must be a number, found: %s' % naics_code
            arrays = []
            for digit in naics_code:
                array = np.zeros(shape=(1,10))
                array[0, int(digit)] = 1
                arrays.append(array)
            outputs.append(np.hstack(arrays))
        return np.vstack(outputs)
    
    def inverse_transform(self, vector):
        assert len(vector.shape) == 2, 'unexpected dimensions for vector'
        output = []
        for row in vector:
            digits = []
            for i in range(6):
                start = i * 10
                end = start + 10
                sub_row = row[start: end]
                digits.append(str(sub_row.argmax()))
            output.append(''.join(digits))
        return output
    
class NAICS1HVectorizer:
    def __init__(self):
        self.tokens = [str(i) for i in range(10)]
    
    def fit(self):
        pass
    
    def transform(self, naics_codes):
        output = np.zeros((len(naics_codes), 6, 10))
        for i, naics_code in enumerate(naics_codes):
            assert len(naics_code) == 6, 'naics codes must be 6 digits'
            for j, digit in enumerate(naics_code):
                output[i, j, int(digit)] = 1
        return output
    
    def inverse_transform(self, vector):
        assert len(vector.shape) == 3, 'input must be a 3 dimensional vector'
        naics_codes = []
        for naics_encoding in vector:
            digits = []
            for position in naics_encoding:
                digits.append(str(position.argmax()))
            naics_code = ''.join(digits)
            naics_codes.append(naics_code)
        return naics_codes
                
        
class WordVectorizerNLTK:
    def __init__(self, max_word_seq, max_words=None, lower=True):
        self.max_word_seq = max_word_seq
        self.lower = lower
        self.max_words = max_words
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.unknown = '__UNKNOWN__'
    
    def fit(self, texts):
        self.counter = collections.defaultdict(int)
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                self.counter[word] += 1
        sort = [(k, self.counter[k]) for k in sorted(self.counter, key=self.counter.get, reverse=True)]
        # We reserve 0's for padding
        self.token_map = {i[0]: n + 1 for n, i in enumerate(sort[:self.max_words])}
        # We add a special unknown token
        self.token_map[self.unknown] = max(self.token_map.values()) + 1
        self.reverse_map = {v: k for k, v in self.token_map.items()}
    
    def transform(self, texts, field=None):
        # Get field specific word sequence, if provided
        if hasattr(self.max_word_seq, 'get'):
            max_word_seq = self.max_word_seq[field]
        else:
            max_word_seq = self.max_word_seq
        # Create output matrix
        n_texts = len(texts)
        output = np.zeros(shape=(n_texts, max_word_seq), dtype=np.int)
        for text_idx, text in enumerate(texts):         
            words = self.word_indexes(text)
            for word_idx, word in enumerate(words[0:max_word_seq]):
                output[text_idx, word_idx] = word
        return output
    
    def inverse_transform(self, vector):
        assert len(vector.shape) == 2, 'input has %s dimensions, expected 2' % len(vector.shape)
        texts = []
        for row in vector:
            words = []
            for index in row:
                if index != 0:
                    words.append(self.reverse_map[index])
            texts.append(' '.join(words))
        return texts
        
    def _tokenize_text(self, text):
        words = self.tokenizer.tokenize(text)
        if self.lower:
            return [word.lower() for word in words]
        else:
            return words
        
    def word_indexes(self, text):
        words = self._tokenize_text(text)
        indexes = []
        for word in words:
            if word in self.token_map:
                index = self.token_map[word]
            else:
                index = self.token_map[self.unknown]
            indexes.append(index)
        return indexes
    

class DummyVectorizer:
    def __init__(self, n, max_n):
        self.n = n
        self.max_n = max_n
    
    def transform(self, rows):
        output = np.zeros(shape=(len(rows), self.max_n))
        output[:, self.n] = 1.0
        return output
    
    def inverse_transform(self, vector):
        return None


class LambdaVectorizer:
    """ Takes a list of feature functions. Each feature function must accept
        a row input and produce a list of features. Intended primarily for
        vectorizing fips_state_code, job_category, and other miscellaneous
        things.
    """
    def __init__(self, functions, tokens=set()):
        self.functions = functions
        self.tokens=tokens
        if tokens:
            self.map_tokens(tokens)
        
    def map_tokens(self, tokens):
        self.token_map = collections.OrderedDict()
        self.reverse_map = {}
        for n, token in enumerate(sorted(self.tokens)):
            self.token_map[token] = n
            self.reverse_map[n] = token
        
    def fit(self, rows):
        self.tokens = set()
        for row in rows:
            for function in self.functions:
                self.tokens.update(function(row))
        self.map_tokens(self.tokens)
            
    def transform(self, rows):
        output = np.zeros(shape=(len(rows), len(self.token_map)))
        for i, row in enumerate(rows):
            for function in self.functions:
                tokens = function(row)
                for token in tokens:
                    token_index = self.token_map[token]
                    output[i, token_index] = 1.0
        return output
    
    def inverse_transform(self, vector):
        output = []
        for row in vector:
            tokens = []
            for index, col in enumerate(row):
                if col != 0:
                    tokens.append(self.reverse_map[index])
        output.append(' '.join(tokens))
        return output
    
def extract_fips(row):
    return ['fips=%s' % row['fips_state_code'].zfill(2)]

def extract_job_category(row):
    categories = [cat for cat in cac_net.constants.JOB_CATEGORIES if row[cat]=='X']
    return ['cat=%s' % cat for cat in categories]


class WordTokenizer():
    """ Tokenizes on sentences, hyphens, pipes, and slashes """
    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.new_punc = [(re.compile(r'-', re.UNICODE), ' \\g<0> '),
                         (re.compile('\|\|', re.UNICODE), ' \\g<0> '),
                         (re.compile(r'/', re.UNICODE), ' \\g<0> '),
                         (re.compile(r'\\', re.UNICODE), ' \\g<0> ')
                         ]
        self.tokenizer.PUNCTUATION.extend(self.new_punc)
    
    def tokenize(self, text):
        words = []
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            nltk_words = self.tokenizer.tokenize(sentence)
            words.extend(nltk_words)
        return words
