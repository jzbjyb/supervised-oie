""" Usage:
    load_pretrained_word_embeddings [--glove=GLOVE_FN]
"""

from docopt import docopt
import numpy as np
from word_index import Word_index
import logging
logging.basicConfig(level = logging.DEBUG)
import sys
sys.path.append("./common")
from symbols import UNK_INDEX, UNK_SYMBOL, UNK_VALUE
from keras.layers import Embedding, SpatialDropout1D, Lambda

from keras_bert import load_trained_model_from_checkpoint
import codecs

class BertEmb:
    '''
    provide the same interface as Glove
    '''
    def __init__(self, bert_config_path, bert_checkpoint_path, bert_dict_path):
        self.bert_config_path = bert_config_path
        self.bert_checkpoint_path = bert_checkpoint_path
        logging.debug('loading bert from {} ...'.format(bert_config_path))
        self.model = load_trained_model_from_checkpoint(
            bert_config_path, bert_checkpoint_path)
        self._load_vocab(bert_dict_path)
        logging.debug('done!')
        self.unk_count = 0
        self.total_count = 0
        self.unk_words = {} # word -> count

    def _load_vocab(self, bert_dict_path):
        self.word_index = {} # bert has [UNK] so we don't need define it
        with codecs.open(bert_dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.word_index[token] = len(self.word_index)

    def get_word_index(self, word, lower=True):
        if lower:
            word = word.lower()
        if word not in self.word_index:
            self.unk_count += 1
            if word not in self.unk_words:
                self.unk_words[word] = 0
            self.unk_words[word] += 1
        self.total_count += 1
        return self.word_index[word] \
            if (word in self.word_index) else self.word_index['[UNK]']

    def get_keras_embedding(self, dropout=0, trainable=False, **kwargs):
        # TODO: trainable is not used
        def crop(dimension, start, end):
            # Crops (or slices) a Tensor on a given dimension from start to end
            # example : to crop tensor x[:, :, 5:10]
            # call slice(2, 5, 10) as you want to crop on the second dimension
            def func(x):
                if dimension == 0:
                    return x[start: end]
                if dimension == 1:
                    return x[:, start: end]
                if dimension == 2:
                    return x[:, :, start: end]
                if dimension == 3:
                    return x[:, :, :, start: end]
                if dimension == 4:
                    return x[:, :, :, :, start: end]
            return Lambda(func)
        return lambda x: SpatialDropout1D(dropout)(crop(1, 1, -1)(self.model(x)))

class Glove:
    """
    Stores pretrained word embeddings for GloVe, and
    outputs a Keras Embeddings layer.
    """
    def __init__(self, fn, dim = None):
        """
        Load a GloVe pretrained embeddings model.
        fn - Filename from which to load the embeddings
        dim - Dimension of expected word embeddings, used as verficiation,
              None avoids this check.
        """
        self.fn = fn
        self.dim = dim
        logging.debug("Loading GloVe embeddings from: {} ...".format(self.fn))
        self._load(self.fn)
        logging.debug("Done!")
        self.unk_count = 0
        self.total_count = 0
        self.unk_words = {} # word -> count

    def _load(self, fn):
        """
        Load glove embedding from a given filename
        """
        self.word_index = {UNK_SYMBOL : UNK_INDEX}
        emb = []
        for line in open(fn):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if self.dim:
                assert(len(coefs) == self.dim)
            else:
                self.dim = len(coefs)

            # Record mapping from word to index
            self.word_index[word] = len(emb) + 1
            emb.append(coefs)

        # Add UNK at the first index in the table
        self.emb = np.array([UNK_VALUE(self.dim)] + emb)
        # Set the vobabulary size
        self.vocab_size = len(self.emb)

    def get_word_index(self, word, lower = True):
        """
        Get the index of a given word (int).
        If word doesnt exists, returns UNK.
        lower - controls whether the word should be lowered before checking map
        """
        if lower:
            word = word.lower()
        if word not in self.word_index:
            self.unk_count += 1
            if word not in self.unk_words:
                self.unk_words[word] = 0
            self.unk_words[word] += 1
        self.total_count += 1
        return self.word_index[word] \
            if (word in self.word_index) else UNK_INDEX

    def get_embedding_matrix(self):
        """
        Return an embedding matrix for use in a Keras Embeddding layer
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        word_index - Maps words in the dictionary to their index (one-hot encoding)
        """
        return self.emb

    def get_keras_embedding(self, **args):
        """
        Get a Keras Embedding layer, loading this embedding as pretrained weights
        The additional arguments given to this function are passed to the Keras Embbeding constructor.
        """
        dp = 0 # no drop out
        if 'dropout' in args:
            dp = args['dropout']
            del args['dropout']
        emb = Embedding(self.vocab_size,self.dim, weights = [self.get_embedding_matrix()], **args)
        return lambda x: SpatialDropout1D(dp)(emb(x))
        '''
        return Embedding(self.vocab_size,
                         self.dim,
                         weights = [self.get_embedding_matrix()],
                         **args)
        '''

if __name__ == "__main__":
    args = docopt(__doc__)
    if args["--glove"] is not None:
        glove_fn = args["--glove"]
        g = Glove(glove_fn)
        emb = g.get_keras_embedding()
    else:
        logging.info(__doc__)
        exit
