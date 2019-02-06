""" Usage:
    model [--train=TRAIN_FN] [--dev=DEV_FN] --test=TEST_FN [--pretrained=MODEL_DIR] [--load_hyperparams=MODEL_JSON] [--saveto=MODEL_DIR] [--restorefrom=RESTORE_DIR]
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # append parent path

import numpy as np
import math
import pandas
import time
from docopt import docopt
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, merge, Bidirectional, Dropout, Reshape, Multiply, Lambda, Add
from keras.layers.merge import concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from load_pretrained_word_embeddings import Glove, BertEmb
from operator import itemgetter
from keras.callbacks import LambdaCallback, ModelCheckpoint
from sklearn import metrics
from pprint import pformat
from common.symbols import SPACY_POS_TAGS
from collections import defaultdict
from parsers.spacy_wrapper import spacy_whitespace_parser as spacy_ws

from keras_bert import load_trained_model_from_checkpoint
from beam_search import TaggingPredict, beamsearch
import keras.backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops

import json
import pdb
from keras.models import model_from_json
import logging
logging.basicConfig(level = logging.DEBUG)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

'''
from tensorflow import set_random_seed
import random
random.seed(2019)
np.random.seed(2019)
set_random_seed(2019)
'''

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    SPE_Y = 2
    def __init__(self, sent_maxlen = None, emb_filename = None,
                 batch_size = 5, seed = 42, sep = '\t',
                 hidden_units = pow(2, 7),trainable_emb = True,
                 emb_dropout = 0.1, num_of_latent_layers = 2,
                 epochs = 10, pred_dropout = 0.1, model_dir = "./models/",
                 classes = None, pos_tag_embedding_size = 5,
                 use_bert = False, model_type='tag', restore_dir=None,
                 save_type='tag',
    ):
        """
        Initialize the model
        model_fn - a model generating function, to be called when
                   training with self as a single argument.
        sent_maxlen - the maximum length in words of each sentence -
                      will be used for padding / truncating
        emb_filename - the filename from which to load the embedding
                       (Currenly only Glove. Idea: parse by filename)
        batch_size - batch size for training
        seed - the random seed for reproduciblity
        sep  - separator in the csv dataset files for this model
        hidden_units - number of hidden units per layer
        trainable_emb - controls if the loss should propagate to the word embeddings during training
        emb_dropout - the percentage of dropout during embedding
        num_of_latent_layers - how many LSTMs to stack
        epochs - the number of epochs to train the model
        pred_dropout - the proportion to dropout before prediction
        model_dir - the path in which to save model
        classes - the classes to be encoded (list of strings)
        pos_tag_embedding_size - The number of features to use when encoding pos tags
        """
        self.model_dir = model_dir
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.sep = sep
        self.encoder = LabelEncoder()
        self.hidden_units = hidden_units
        self.emb_filename = emb_filename
        if use_bert:
            bert_root = '/home/zhengbaj/exp/bert_pretrained/uncased_L-12_H-768_A-12/'
            self.emb = BertEmb(bert_config_path=bert_root + 'bert_config.json', 
                bert_checkpoint_path=bert_root + 'bert_model.ckpt', 
                bert_dict_path=bert_root + 'vocab.txt')
        else:
            self.emb = Glove(emb_filename)
        #self.embedding_size = self.emb.dim
        self.trainable_emb = trainable_emb
        self.emb_dropout = emb_dropout
        self.num_of_latent_layers = num_of_latent_layers
        self.epochs = epochs
        self.pred_dropout = pred_dropout
        self.classes = classes
        self.pos_tag_embedding_size = pos_tag_embedding_size
        self.use_bert = use_bert
        # 1. 'tag' for tagging model
        # 2. 'conf' for confidence model based on tagging model
        # 3. 'conf_new' for confidence model based on new model
        self.model_type = model_type
        if self.model_type not in {'tag', 'conf', 'conf_new'}:
            raise ValueError('model_type not supported')
        self.restore_dir = restore_dir # path to the weights from which the model is restored

        # based on model_type and restore_dir,
        # choose model construction function and training function
        if self.model_type == 'tag':
            if self.restore_dir is None: # train new tagging model
                self.model_fn = self.set_vanilla_model
            else: # load pretrained model
                self.model_fn = self.set_model_from_file
            self.train = self.train_tagging_model
        elif self.model_type == 'conf':
            self.alpha = 1.0 # control trade-off between mle and hinge loss
            self.model_fn = self.set_confidence_pointwise_model
            self.train = self.train_confidence_pointwise_model
            # save_type decides which model to save
            # 1. 'tag' is to load and save the tagging model
            # 2. 'conf' is to load and save the confidence model
            # 3. 'tag_conf' is to load tagging model and save confidence model
            self.save_type = save_type
            if self.save_type not in {'tag', 'conf', 'tag_conf'}:
                raise ValueError('save_type not supported')

        np.random.seed(self.seed)

    def get_callbacks(self, X, model=None):
        """
        Sets these callbacks as a class member.
        X is the encoded dataset used to print a sample of the output.
        Callbacks created:
        1. Sample output each epoch
        2. Save best performing model each epoch
        """
        model_output = LambdaCallback(on_epoch_end=lambda epoch, logs: \
            logging.debug(model.predict(X)))
        sample_output = LambdaCallback(on_epoch_end=lambda epoch, logs: \
            logging.debug(pformat(self.sample_labels(self.model.predict(X)))))
        checkpoint = ModelCheckpoint(os.path.join(self.model_dir, 'weights.hdf5'),
            verbose=1, save_best_only=False) # TODO: is there a way to save by best val_acc?

        return [checkpoint]
        #return [model_output]

    def plot(self, fn, train_fn):
        """
        Plot this model to an image file
        Train file is needed as it influences the dimentions of the RNN
        """
        from keras.utils.visualize_util import plot
        X, Y = self.load_dataset(train_fn)
        self.model_fn()
        plot(self.model, to_file = fn)

    def classes_(self):
        """
        Return the classes which are classified by this model
        """
        if self.classes is not None:
            return self.classes
        try:
            return self.encoder.classes_
        except:
            return None
        '''
        try:
            return self.encoder.classes_
        except:
            return self.classes
        '''

    def train_and_test(self, train_fn, test_fn):
        """
        Train and then test on given files
        """
        logging.info("Training..")
        self.train(train_fn)
        logging.info("Testing..")
        return self.test(test_fn)
        logging.info("Done!")

    def train_tagging_model(self, train_fn, dev_fn):
        """
        Train this model on a given train dataset
        Dev test is used for model checkpointing
        """
        # load dataset
        X_train, Y_train = self.load_dataset(train_fn)
        X_dev, Y_dev = self.load_dataset(dev_fn)
        logging.debug('UNK appears {} times out of {} ({}%)'.format(
            self.emb.unk_count, self.emb.total_count, 
            self.emb.unk_count * 100 / (self.emb.total_count + 1e-5)))
        top_unk_words = sorted(self.emb.unk_words.items(), key=lambda x: -x[1])[:10]
        logging.debug('totally {} UNK words: {}'.format(len(self.emb.unk_words), top_unk_words))
        logging.debug("Classes: {}".format((self.num_of_classes(), self.classes_())))

        # Set model params, called here after labels have been identified in load dataset
        self.model_fn()

        # Create a callback to print a sample after each epoch
        logging.debug("Training model on {}".format(train_fn))
        self.model.fit(X_train, Y_train,
                       batch_size = self.batch_size,
                       epochs = self.epochs,
                       validation_data = (X_dev, Y_dev),
                       sample_weight=X_train['mask_inputs'],
                       callbacks = self.get_callbacks(X_train))

    def tag_model_to_sentence_score(self, tag_inputs, mask_inputs, tag_true):
        # SHAPE: (None, sent_maxlen, num_class)
        tag_prob = self.model(tag_inputs)
        # clip prob
        def clip(prob):
            epsilon_ = ops.convert_to_tensor(K.epsilon(), dtype='float32')
            prob = clip_ops.clip_by_value(prob, epsilon_, 1. - epsilon_)
            return prob
        tag_prob = Lambda(clip)(tag_prob)
        # log prob
        tag_prob = Lambda(lambda x: K.log(x))(tag_prob)
        # SHAPE: (None, sent_maxlen)
        cast_to_float32 = Lambda(lambda x: K.cast(x, 'float32'))
        tag_prob = Multiply()([cast_to_float32(tag_true), tag_prob])
        # SHAPE: (None, sent_maxlen)
        tag_prob = Lambda(lambda x: K.sum(x, axis=-1))(tag_prob)
        reshape_sent = Lambda(lambda x: K.reshape(x, (-1, self.max_subsent * self.sent_maxlen)))
        # SHAPE: (None, max_subsent * sent_maxlen)
        mask_inputs = cast_to_float32(reshape_sent(mask_inputs))
        # SHAPE: (None, max_subsent * sent_maxlen)
        tag_prob = reshape_sent(tag_prob)
        # SHAPE: (None, max_subsent * sent_maxlen)
        tag_prob = Multiply()([mask_inputs, tag_prob])
        # SHAPE: (None,)
        mask_inputs = Lambda(lambda x: K.sum(x, axis=-1))(mask_inputs)
        # SHAPE: (None,)
        tag_prob_sum = Lambda(lambda x: K.sum(x, axis=-1))(tag_prob) # sum log prob
        # SHAPE: (None,)
        tag_prob_avg = Lambda(lambda x: x[0] / x[1])([tag_prob_sum, mask_inputs]) # avg log prob
        tag_prob_sum_norm = Lambda(lambda x: x[0] / K.mean(x[1]))(
            [tag_prob_sum, mask_inputs]) # sum log prob normalized by average seq len
        return tag_prob_avg, tag_prob_sum_norm

    def mle_hinge_loss(self):
        '''
        Final loss is alpha * mle + hinge
        '''
        def cus_loss(y_true, y_pred):
            ms = K.cast(K.equal(y_true, RNN_model.SPE_Y), dtype=y_pred.dtype) # mle samples
            hs = 1 - ms # hinge samples
            # mle loss (nll loss)
            mle = -K.mean(y_pred * ms, axis=-1) # y_pred is sum of log prob
            # hinge loss
            hl =  K.mean(K.maximum(1. - y_true * y_pred, 0.) * hs, axis=-1)
            loss = self.alpha * mle + hl
            return loss
        return cus_loss

    def set_confidence_pointwise_model(self):
        # sentence-level inputs and subsentence-level input shapes
        sent_maxlen = self.max_subsent * self.sent_maxlen
        if self.use_bert:
            inputs = \
                [Input(shape=(sent_maxlen + self.max_subsent * 2,), dtype="int32", name="word_inputs"),
                 Input(shape=(sent_maxlen + self.max_subsent * 2,), dtype="int32", name="word_segment_inputs")]
            new_shapes = [(self.sent_maxlen + 2,), (self.sent_maxlen + 2,)]
        else:
            inputs = [Input(shape=(sent_maxlen,), dtype="int32", name="word_inputs")]
            new_shapes = [(self.sent_maxlen,)]
        inputs.append(Input(shape=(sent_maxlen,), dtype="int32", name="predicate_inputs"))
        new_shapes.append((self.sent_maxlen,))
        inputs.append(Input(shape=(sent_maxlen,), dtype="int32", name="postags_inputs"))
        new_shapes.append((self.sent_maxlen,))
        inputs.append(Input(shape=(sent_maxlen,), dtype="int32", name="mask_inputs"))
        new_shapes.append((self.sent_maxlen,))
        inputs.append(Input(shape=(sent_maxlen, self.num_of_classes()), dtype="int32", name="tag_true"))
        new_shapes.append((self.sent_maxlen, self.num_of_classes()))

        # get subsentence-level inputs
        sub_inputs = [Lambda(lambda x, sh: K.reshape(x, (-1,) + sh), arguments={'sh': ns})(inp) for inp, ns in
                      zip(inputs, new_shapes)]
        tag_true = sub_inputs[-1]
        mask_inputs = sub_inputs[-2]
        sub_inputs = sub_inputs[:-2]

        # sentence-level inputs
        loss_ind = Input(shape=(1,), dtype="int32", name="loss_ind") # indicate whether to use mle loss
        inputs.append(loss_ind)
        loss_ind = Lambda(lambda x: K.cast(K.equal(x, RNN_model.SPE_Y), dtype='float32'))(loss_ind)

        # build or load original tagging model
        if self.restore_dir is None:
            self.set_vanilla_model(dump_json=False)
        elif self.save_type == 'tag' or self.save_type == 'tag_conf':
            self.set_model_from_file()
        elif self.save_type == 'conf':
            self.set_vanilla_model(dump_json=False)

        # get sentence score
        avg_log_prob, sum_log_prob = self.tag_model_to_sentence_score(sub_inputs, mask_inputs, tag_true)
        avg_log_prob = Reshape((1,))(avg_log_prob)
        sum_log_prob = Reshape((1,))(sum_log_prob)

        # get output for loss
        avg_log_prob = Dense(1, activation=None, use_bias=True)(avg_log_prob)  # linear layer
        output = Multiply()([sum_log_prob, loss_ind])
        output = Add()([output, Multiply()([avg_log_prob, Lambda(lambda x: 1-x)(loss_ind)])])

        # build and compile model
        new_model = Model(input=inputs, output=[output])
        #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.00001)
        #new_model.compile(optimizer=opt, loss='hinge', metrics=['hinge'])
        mh_loss = self.mle_hinge_loss()
        new_model.compile(optimizer='adam', loss=mh_loss, metrics=[mh_loss])
        if self.restore_dir is not None and self.save_type == 'conf':
            new_model.load_weights(os.path.join(self.restore_dir, 'weights.hdf5'))
        else:
            new_model.summary()
        return new_model

    def train_confidence_pointwise_model(self, train_fn, dev_fn):
        # load train dataset (must load data before building the model)
        X_train, Y_train = self.load_dataset(train_fn, pad_sent=True)
        X_train['tag_true'] = Y_train
        X_train = self.dataset_to_sentence_level(X_train)
        Y_train = self.load_dataset_y(train_fn, mode='hinge')
        X_train['loss_ind'] = Y_train
        X_weight = self.load_dataset_weight(train_fn)
        # load dev dataset
        X_dev, Y_dev = self.load_dataset(dev_fn, pad_sent=True)
        X_dev['tag_true'] = Y_dev
        X_dev = self.dataset_to_sentence_level(X_dev)
        Y_dev = self.load_dataset_y(dev_fn, mode='hinge')
        X_dev['loss_ind'] = Y_dev

        new_model = self.model_fn()

        # train the model
        def save_model():
            #self.save_model_to_file(os.path.join(self.model_dir, "model.json"))
            if self.save_type == 'tag':
                self.model.save_weights(os.path.join(self.model_dir, 'weights.hdf5'))
            elif self.save_type == 'conf' or self.save_type == 'tag_conf':
                new_model.save_weights(os.path.join(self.model_dir, 'weights.hdf5'))
        callback = LambdaCallback(on_epoch_end=lambda epoch, logs: save_model()) # only save tagging model
        new_model.fit(X_train, Y_train, batch_size=self.batch_size,
                      sample_weight=X_weight,
                      epochs=self.epochs,
                      callbacks=[callback],
                      validation_data=(X_dev, Y_dev))

    def train_confidence_pairwise_model(self, train_fn, dev_fn):
        '''
        both train_fn and dev_fn contain pair of samples
        '''
        # load the dataset (must load data before building the model)
        X_train, Y_train = self.load_dataset(train_fn, pad_sent=True)
        X_train['tag_true'] = Y_train
        Y_train = np.ones((Y_train.shape[0])) # useless, just a placeholder
        X_dev, Y_dev = self.load_dataset(dev_fn, pad_sent=True)
        X_dev['tag_true'] = Y_dev
        Y_dev = np.ones((Y_dev.shape[0])) # useless, just a placeholder

        # build original tagging model
        self.set_vanilla_model()

        # get margin score
        tag_prob = self.model.output
        mask_inputs = self.model.input[-1]
        tag_true = Input(shape=(self.sent_maxlen, self.num_of_classes()),
                         dtype="int32", name="tag_true") # additional input for margin model
        cast_to_float32 = Lambda(lambda x: K.cast(x, 'float32'))
        tag_prob = Multiply()([cast_to_float32(tag_true), Lambda(lambda x: K.log(x))(tag_prob)])
        tag_prob = Lambda(lambda x: K.sum(x, axis=-1))(tag_prob)
        reshape_sent = Lambda(lambda x: K.reshape(x, (-1, self.max_subsent * self.sent_maxlen)))
        mask_inputs = cast_to_float32(reshape_sent(mask_inputs))
        tag_prob = reshape_sent(tag_prob)
        tag_prob = Multiply()([mask_inputs, tag_prob])
        mask_inputs = Lambda(lambda x: K.sum(x, axis=-1))(mask_inputs)
        tag_prob = Lambda(lambda x: K.sum(x, axis=-1))(tag_prob)
        sent_score = Lambda(lambda x: x[0] / x[1])([tag_prob, mask_inputs])
        margin_score = Lambda(lambda x: x[::2] - x[1::2])(sent_score)
        margin_score = Lambda(lambda x: K.reshape(x, (-1, 1)))(margin_score)
        zeros_like = Lambda(lambda x: K.zeros_like(x))
        margin_score_padded = concatenate(
            [margin_score] + [zeros_like(margin_score)] * (self.max_subsent * 2 - 1), axis=0)

        # build and compile model
        new_model = Model(input=self.model.input + [tag_true], output=[margin_score_padded])
        new_model.compile(optimizer='adam', loss='hinge', metrics=['hinge'])
        new_model.summary()

        # train the model
        # batch_size requires a multiple of max_subsent
        new_model.fit(X_train, Y_train, batch_size=self.batch_size * self.max_subsent,
                      epochs=self.epochs, validation_data=(X_dev, Y_dev),)
                      #callbacks=self.get_callbacks(X_train))

    @staticmethod
    def to_categorical(*args, **kwargs):
        if len(args[0]) == 0:
            return []
        return np_utils.to_categorical(*args, **kwargs)

    @staticmethod
    def consolidate_labels(labels):
        """
        Return a consolidated list of labels, e.g., O-A1 -> O, A1-I -> A
        """
        return map(RNN_model.consolidate_label , labels)

    @staticmethod
    def consolidate_label(label):
        """
        Return a consolidated label, e.g., O-A1 -> O, A1-I -> A
        """
        return label.split("-")[0] if label.startswith("O") else label

    def predict_sentence_beamsearch(self, sent, k=1):
        '''
        k is the number of extractions generated for each predicate
        '''
        ret = []
        sent_str = " ".join(sent)

        # Extract predicates by looking at verbal POS
        preds = [(word.i, str(word))
                 for word
                 in spacy_ws(sent_str)
                 if word.tag_.startswith("V")]

        classes = self.classes_()

        for ind, pred in preds:
            cur_sample = self.create_sample(sent, ind)
            X = self.encode_inputs([cur_sample], get_output=False)
            predict_prob = self.model.predict(X)
            predict_prob = predict_prob.reshape(-1, predict_prob.shape[-1])
            mask = X['mask_inputs'].flatten()
            assert len(predict_prob) == len(mask), 'predication results not in the same shape as mask'
            predict_prob = np.array([p for p,m in zip(predict_prob, mask) if m == 1])
            tp = TaggingPredict(predict_prob)
            samples, scores = beamsearch(tp, k=k)
            for sample, score in zip(samples, scores):
                ret.append(((ind, pred),
                            [(self.consolidate_label(classes[label]), prob) for label, prob in sample]))
        return ret

    def predict_sentence(self, sent):
        """
        Return a predicted label for each word in an arbitrary length sentence
        sent - a list of string tokens
        """
        ret = []
        sent_str = " ".join(sent)

        # Extract predicates by looking at verbal POS
        preds = [(word.i, str(word))
                 for word
                 in spacy_ws(sent_str)
                 if word.tag_.startswith("V")]

        # Calculate num of samples (round up to the nearst multiple of sent_maxlen)
        num_of_samples = np.ceil(float(len(sent)) / self.sent_maxlen) * self.sent_maxlen # TODO: python2 divide bug
        num_of_samples = int(num_of_samples)

        # Run RNN for each predicate on this sentence
        for ind, pred in preds:
            cur_sample = self.create_sample(sent, ind)
            X = self.encode_inputs([cur_sample], get_output=False)
            results = self.transform_output_probs(self.model.predict(X), get_prob=True).reshape(-1, 2)
            mask = X['mask_inputs'].flatten()
            assert len(results) == len(mask), 'predication results not in the same shape as mask'
            results = zip(results, mask) # use mask to get the label (no truncation needed)
            ret.append(((ind, pred),
                        [(self.consolidate_label(label), float(prob))
                         for ((label, prob), mask_label) in results if mask_label == 1]))
            '''
            ret.append(((ind, pred),
                        [(self.consolidate_label(label), float(prob))
                         for (label, prob) in
                         self.transform_output_probs(self.model.predict(X),           # "flatten" and truncate
                                                     get_prob = True).reshape(num_of_samples,
                                                                              2)[:len(sent)]]))
            '''
        return ret

    def create_sample(self, sent, head_pred_id):
        """
        Return a dataframe which could be given to encode_inputs
        """
        return pandas.DataFrame({"word": sent,
                                 "run_id": [-1] * len(sent), # Mock running id
                                 "head_pred_id": head_pred_id})

    def test_confidence_model(self, test_fn, test_extraction_fn, new_extractio_fn):
        # load test dataset
        X_test, Y_test = self.load_dataset(test_fn, pad_sent=True)
        X_test['tag_true'] = Y_test
        X_test = self.dataset_to_sentence_level(X_test)
        Y_test = self.load_dataset_y(test_fn, mode='hinge')
        X_test['loss_ind'] = np.zeros_like(Y_test)

        new_model = self.model_fn()

        # test the model
        y = new_model.predict(X_test)
        y = y.flatten()
        with open(test_extraction_fn, 'r') as fin, open(new_extractio_fn, 'w') as fout:
            for i, l in enumerate(fin):
                l = l.split('\t')
                l[1] = str(y[i])
                fout.write('\t'.join(l))
        return y

    def test_tagging_model(self, test_fn, eval_metrics):
        """
        Evaluate this model on a test file
        eval metrics is a list composed of:
        (name, f: (y_true, y_pred) -> float (some performance metric))
        Prints and returns the metrics name and numbers
        """
        # Load gold and predict
        X, Y = self.load_dataset(test_fn)
        y = self.model.predict(X)

        # Get most probable predictions and flatten
        Y = RNN_model.consolidate_labels(self.transform_output_probs(Y).flatten())
        y = RNN_model.consolidate_labels(self.transform_output_probs(y).flatten())

        # Run evaluation metrics and report
        # TODO: is it possible to compare without the padding?
        ret = []
        for (metric_name, metric_func) in eval_metrics:
            ret.append((metric_name, metric_func(Y, y)))
            logging.debug("calculating {}".format(ret[-1]))

        for (metric_name, metric_val) in ret:
            logging.info("{}: {:.4f}".format(metric_name,
                                             metric_val))
        return Y, y, ret

    def load_dataset_weight(self, fn):
        df = pandas.read_csv(fn, sep=self.sep, header=0, keep_default_na=False, quoting=3)
        sents = self.get_sents_from_df(df)
        w = np.array([float(sent.weight.values[0]) for sent in sents])
        return w

    def load_dataset_y(self, fn, mode='default'):
        df = pandas.read_csv(fn, sep=self.sep, header=0, keep_default_na=False, quoting=3)
        sents = self.get_sents_from_df(df)
        if mode == 'default':
            y = np.array([int(sent.y.values[0]) for sent in sents])
        elif mode == 'hinge': # +1 -1
            y = np.array([(lambda x: x if x != 0 else -1)(int(sent.y.values[0])) for sent in sents])
        return y.reshape((-1, 1))

    def dataset_to_sentence_level(self, X):
        def to_new_shape(x):
            shape = x.shape
            if shape[0] % self.max_subsent != 0:
                raise  Exception('input does not constitute valid sentence')
            new_shape =  (shape[0] // self.max_subsent,) + (shape[1] * self.max_subsent,) + shape[2:]
            return x.reshape(new_shape)
        return dict((k, to_new_shape(X[k])) for k in X)

    def load_dataset(self, fn, pad_sent=False):
        """
        Load a supervised OIE dataset from file
        If pad_sent=True, we split all the sentence into the same numbers of subsentence
        """
        df = pandas.read_csv(fn, sep=self.sep, header=0, keep_default_na=False, quoting=3)

        # Encode one-hot representation of the labels
        if self.classes_() is None:
            self.encoder.fit(df.label.values)

        # Split according to sentences and encode
        sents = self.get_sents_from_df(df)
        logging.info("{} has {} sentences".format(fn, len(sents)))
        X, Y = self.encode_inputs(sents, get_output=True, pad_sent=pad_sent)
        return (X, Y)

    def get_sents_from_df(self, df):
        dfv = df.values
        col = df.columns.values.tolist()
        ridx = col.index('run_id')
        df_li = []
        last_idx = 0
        for i in range(1, len(dfv)):
            if dfv[i, ridx] != dfv[i-1, ridx]:
                sdf = pandas.DataFrame(dfv[last_idx:i], columns=df.columns)
                df_li.append(sdf)
                last_idx = i
        if len(dfv) > 0:
            sdf = pandas.DataFrame(dfv[last_idx:], columns=df.columns)
            df_li.append(sdf)
        return df_li

    def get_sents_from_df_slow(self, df):
        """
        Split a data frame by rows accroding to the sentences
        """
        return [df[df.run_id == run_id]
                for run_id
                in sorted(set(df.run_id.values))]

    def get_fixed_size(self, sents, pad_sent=False):
        """
        Partition sents into lists of sent_maxlen elements
        (execept the last in each sentence, which might be shorter)
        """
        if pad_sent and not hasattr(self, 'max_subsent'):
            # split sentence into equal numbers of subsentence
            self.max_subsent = np.max([np.ceil(len(sent) / self.sent_maxlen) for sent in sents]) # TODO: python2 divide bug
            self.max_subsent = int(self.max_subsent)
            logging.debug('max_subsent is {}'.format(self.max_subsent))
        if not pad_sent:
            return [sent[s_ind : s_ind + self.sent_maxlen]
                    for sent in sents
                    for s_ind in range(0, len(sent), self.sent_maxlen)]
        else:
            return [sent[s_ind * self.sent_maxlen : s_ind * self.sent_maxlen + self.sent_maxlen]
                    for sent in sents
                    for s_ind in range(self.max_subsent)]

    def get_head_pred_word(self, full_sent):
        """
        Get the head predicate word from a full sentence conll.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]

    def get_head_pred_word_pos(self, full_sent):
        pred_ind = full_sent.head_pred_id.values[0]
        if pred_ind != -1:
            return pred_ind
        pred_head_word = full_sent.pred.values[0].split(' ')[0]
        for i, word in enumerate(full_sent.word.values):
            if word == pred_head_word:
                return i
        raise Exception('can\'t find predicate haed')

    def encode_inputs(self, sents, get_output=False, pad_sent=False):
        """
        Given a dataframe which is already split to sentences,
        encode inputs for rnn classification.
        Should return a dictionary of sequences of sample of length maxlen.
        """
        word_inputs = []
        pred_inputs = []
        pos_inputs = []
        mask_inputs = []
        label_outputs = []

        # Preproc to get all preds per run_id
        # Sanity check - make sure that all sents agree on run_id
        assert(all([len(set(sent.run_id.values)) == 1
                    for sent in sents]))
        run_id_to_pred = dict([(int(sent.run_id.values[0]),
                                self.get_head_pred_word(sent))
                               for sent in sents])

        # Construct a mapping from running word index to pos
        word_id_to_pos = {}

        # convert dataframe to list and tokenize sentence
        sent_word_list = []
        sent_pos_list = []
        sent_label_list = []
        sent_pred_list = []
        sent_mask_list = []
        sent_runid_list = []
        for sent in sents:
            indices = sent.index.values.tolist()
            words = sent.word.values.tolist()
            if get_output:
                labels = self.transform_labels(sent.label.values.tolist())
            else:
                labels = [0] * len(words) # fake labels
            preds = [0] * len(words)
            preds[self.get_head_pred_word_pos(sent)] = 1
            # get pos
            for index, word in zip(indices, spacy_ws(" ".join(words))):
                word_id_to_pos[index] = word.tag_
            poss = [word_id_to_pos[index] for index in indices] # poss has the same length as words
            # tokenize (mainly for bert with use sub words)
            new_words, mask = self.emb.tokenize(words)
            # convert poss
            new_poss, new_labels, new_preds = [], [], []
            for pos_tag, lab, pr in zip(poss, labels, preds):
                si = len(new_poss)
                while mask[si] != 1:
                    si += 1
                    new_poss.append(pos_tag)
                    new_labels.append(lab)
                    new_preds.append(pr)
                new_poss.append(pos_tag)
                new_labels.append(lab)
                new_preds.append(pr)
            # store all lists
            sent_word_list.append(new_words)
            sent_pos_list.append(new_poss)
            sent_label_list.append(new_labels)
            sent_pred_list.append(new_preds)
            sent_mask_list.append(mask)
            sent_runid_list.append([sent.run_id.values[0]] * len(mask))

        fixed_size_sent_word_list = self.get_fixed_size(sent_word_list, pad_sent=pad_sent)
        fixed_size_sent_pos_list = self.get_fixed_size(sent_pos_list, pad_sent=pad_sent)
        fixed_size_sent_label_list = self.get_fixed_size(sent_label_list, pad_sent=pad_sent)
        fixed_size_sent_pred_list = self.get_fixed_size(sent_pred_list, pad_sent=pad_sent)
        fixed_size_sent_mask_list = self.get_fixed_size(sent_mask_list, pad_sent=pad_sent)
        fixed_size_sent_runid_list = self.get_fixed_size(sent_runid_list, pad_sent=pad_sent)

        for sent_word, sent_pos, sent_label, sent_pred, sent_mask, sent_runid in zip(
            fixed_size_sent_word_list, fixed_size_sent_pos_list, 
            fixed_size_sent_label_list, fixed_size_sent_pred_list,
            fixed_size_sent_mask_list, fixed_size_sent_runid_list):
            if self.use_bert:
                # add special symbol
                # For sentence shorted than sent_maxlen, [SEP] symbol is hard to remove
                # from the output of bert. Hopefully, this is not a big problem.
                sent_word.insert(0, '[CLS]')
                sent_word.append('[SEP]')
            pos_tags_encodings = \
                [SPACY_POS_TAGS.index(pt) if pt in SPACY_POS_TAGS else 0 for pt in sent_pos]
            word_encodings = [self.emb.get_word_index(w, lower=False) for w in sent_word]
            if self.use_bert:
                # in bert, the embedding of predicate should be selected from the output of bert
                # so we feed the mask as input for selection purpose
                pred_word_encodings = [pr for pr in sent_pred]
            else:
                if len(sent_runid) > 0:
                    pred_word = run_id_to_pred[int(sent_runid[0])]
                    pred_word_encodings = [self.emb.get_word_index(pred_word) for _ in sent_word]
                else:
                    pred_word_encodings = []

            word_inputs.append([Sample(w) for w in word_encodings])
            pred_inputs.append([Sample(w) for w in pred_word_encodings])
            pos_inputs.append([Sample(pos) for pos in pos_tags_encodings])
            mask_inputs.append([Sample(m) for m in sent_mask])
            label_outputs.append(list(RNN_model.to_categorical(sent_label, num_classes=self.num_of_classes())))

        # Pad / truncate to desired maximum length
        ret = defaultdict(lambda: [])

        input_titles = ["word_inputs", "predicate_inputs", "postags_inputs", "mask_inputs"]
        input_tensors = [word_inputs, pred_inputs, pos_inputs, mask_inputs]
        if self.use_bert:
            input_pad_lens = [self.sent_maxlen+2, self.sent_maxlen, self.sent_maxlen, self.sent_maxlen]
        else:
            input_pad_lens = [self.sent_maxlen, self.sent_maxlen, self.sent_maxlen, self.sent_maxlen]
        for name, sequence, pad_len in zip(input_titles, input_tensors, input_pad_lens):
            for samples in pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = pad_len):
                ret[name].append([sample.encode() for sample in samples])
        input_data = {k: np.array(v) for k, v in ret.iteritems()}
        if self.use_bert:
            # bert hinput
            input_data['word_segment_inputs'] = np.zeros_like(input_data['word_inputs'])
        if not get_output:
            return input_data
        output_data = np.ndarray(shape=(len(label_outputs), self.sent_maxlen, self.num_of_classes()),
            buffer = np.array(pad_sequences(
                label_outputs, lambda : np.zeros(self.num_of_classes()), maxlen = self.sent_maxlen)))
        return input_data, output_data

    def encode_outputs(self, sents, pad_sent=False):
        """
        Deprecated
        Given a dataframe split to sentences, encode outputs for rnn classification.
        Should return a list sequence of sample of length maxlen.
        """
        output_encodings = []
        sents = self.get_fixed_size(sents, pad_sent=pad_sent)
        # Encode outputs
        for sent in sents:
            output_encodings.append(list(np_utils.to_categorical(list(self.transform_labels(sent.label.values)),
                                                                 num_classes = self.num_of_classes())))

        # Pad / truncate to maximum length
        return np.ndarray(shape = (len(sents),
                                  self.sent_maxlen,
                                  self.num_of_classes()),
                          buffer = np.array(pad_sequences(output_encodings,
                                                          lambda : \
                                                            np.zeros(self.num_of_classes()),
                                                          maxlen = self.sent_maxlen)))

    def transform_labels(self, labels):
        """
        Encode a list of textual labels
        """
        # Fallback:
        # return self.encoder.transform(labels)
        classes  = list(self.classes_())
        def index_with_exception(label):
            try:
                return classes.index(label)
            except ValueError:
                return classes.index('O')
        return [index_with_exception(label) for label in labels]

    def transform_output_probs(self, y, get_prob = False):
        """
        Given a list of probabilities over labels, get the textual representation of the
        most probable assignment
        """
        return np.array(self.sample_labels(y,
                                  num_of_sents = len(y), # all sentences
                                  num_of_samples = max(map(len, y)), # all words
                                  num_of_classes = 1, # Only top probability
                                  start_index = 0, # all sentences
                                  get_prob = get_prob, # Indicate whether to get only labels
        ))

    def inverse_transform_labels(self, indices):
        """
        Encode a list of textual labels
        """
        classes = self.classes_()
        return [classes[ind] for ind in indices]

    def num_of_classes(self):
        """
        Return the number of ouput classes
        """
        return len(self.classes_())

    # Functional Keras -- all of the following are currying functions expecting models as input
    # https://keras.io/getting-started/functional-api-guide/

    def embed_word(self):
        """
        Embed word sequences using self's embedding class
        """
        return self.emb.get_keras_embedding(dropout = self.emb_dropout,
                                            trainable = self.trainable_emb,
                                            input_length = self.sent_maxlen)

    def embed_pos(self):
        """
        Embed Part of Speech using this instance params
        """
        return Embedding(output_dim = self.pos_tag_embedding_size,
                         input_dim = len(SPACY_POS_TAGS),
                         input_length = self.sent_maxlen)

    def predict_classes(self):
        """
        Predict to the number of classes
        Named arguments are passed to the keras function
        """
        return lambda x: self.stack(x,
                                    [lambda : TimeDistributed(Dense(output_dim = self.num_of_classes(),
                                                                    activation = "softmax"))] +
                                    [lambda : TimeDistributed(Dense(self.hidden_units,
                                                                    activation='relu'))] * 3)
    def stack_latent_layers(self, n):
        """
        Stack n bidi LSTMs
        """
        return lambda x: self.stack(x, [lambda : Bidirectional(LSTM(self.hidden_units,
                                                                    return_sequences = True))] * n )

    def stack(self, x, layers):
        """
        Stack layers (FIFO) by applying recursively on the output,
        until returing the input as the base case for the recursion
        """
        if not layers:
            return x # Base case of the recursion is the just returning the input
        else:
            return layers[0]()(self.stack(x, layers[1:]))

    def set_model_from_file(self, rebuild=False):
        """
        Receives an instance of RNN and returns a model from the self.model_dir
        path which should contain a file named: model.json,
        and a single file with the hdf5 extension.
        Note: Use this function for a pretrained model, running model training
        on the loaded model will override the files in the model_dir
        """
        from glob import glob

        weights_fn = glob(os.path.join(self.restore_dir, "*.hdf5"))
        assert len(weights_fn) == 1, \
            "More/Less than one weights file in {}: {}".format(self.restore_dir, weights_fn)
        weights_fn = weights_fn[0]
        model_json_fn = os.path.join(self.restore_dir, "./model.json")

        try:
            if rebuild:
                raise Exception('rebuild model')
            self.model = model_from_json(open(model_json_fn).read())
            self.model.compile(optimizer="adam", loss='categorical_crossentropy',
                sample_weight_mode='temporal', metrics=["accuracy"])
            logging.debug('use json file to build model')
        except:
            self.set_vanilla_model(dump_json=False)
            logging.debug('use function to build model')
        logging.debug('load weights from {}'.format(weights_fn))
        self.model.load_weights(weights_fn)

    def set_vanilla_model(self, dump_json=True):
        """
        Set a Keras model for predicting OIE as a member of this class
        Can be passed as model_fn to the constructor
        """
        logging.debug("Setting vanilla model")
        # Build model

        ## Embedding Layer
        word_embedding_layer = self.embed_word()
        if self.use_bert:
            # the embedding of predicate should come from bert's output
            predicate_embedding_layer = \
                self.emb.get_transformed_embedding(
                    input_length=self.sent_maxlen, dropout=self.emb_dropout)
        else:
            predicate_embedding_layer = word_embedding_layer
        pos_embedding_layer = self.embed_pos()

        ## Deep layers
        latent_layers = self.stack_latent_layers(self.num_of_latent_layers)

        ## Dropout
        dropout = Dropout(self.pred_dropout)

        ## Prediction
        predict_layer = self.predict_classes()

        ## Prepare input features, and indicate how to embed them
        if self.use_bert:
            inputs_and_embeddings = \
                [([Input(shape=(self.sent_maxlen+2,), dtype="int32", name="word_inputs"),
                   Input(shape=(self.sent_maxlen+2,), dtype="int32", name="word_segment_inputs")],
                word_embedding_layer),
                (Input(shape=(self.sent_maxlen,), dtype="int32", name="predicate_inputs"),
                predicate_embedding_layer)]
        else:
            inputs_and_embeddings = \
                [(Input(shape=(self.sent_maxlen,), dtype="int32", name="word_inputs"),
                word_embedding_layer),
                (Input(shape=(self.sent_maxlen,), dtype="int32", name="predicate_inputs"), 
                predicate_embedding_layer)]
        inputs_and_embeddings.append(
            (Input(shape=(self.sent_maxlen,), dtype="int32", name = "postags_inputs"),
            pos_embedding_layer))
        lstm_inputs = [embed(inp) for inp, embed in inputs_and_embeddings]

        ## Concat all inputs and run on deep network
        output = predict_layer(dropout(latent_layers(concatenate(lstm_inputs, axis=-1))))

        # Build model
        model_input = []
        for inps, embed in inputs_and_embeddings:
            if type(inps) is not list:
                inps = [inps]
            model_input.extend(inps)
        # mask inputs
        #model_input.append(Input(shape=(self.sent_maxlen,), dtype="int32", name = "mask_inputs"))
        self.model = Model(input = model_input, output = [output])

        # Loss
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           sample_weight_mode='temporal',
                           metrics=['categorical_accuracy'])

        ## Dump model
        if dump_json:
            self.model.summary()
            # Save model json to file
            self.save_model_to_file(os.path.join(self.model_dir, "model.json"))

    def to_json(self):
        """
        Encode a json of the parameters needed to reload this model
        """
        return {
            "sent_maxlen": self.sent_maxlen,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "sep": self.sep,
            "classes": list(self.classes_()),
            "hidden_units": self.hidden_units,
            "trainable_emb": self.trainable_emb,
            "emb_dropout": self.emb_dropout,
            "num_of_latent_layers": self.num_of_latent_layers,
            "epochs": self.epochs,
            "pred_dropout": self.pred_dropout,
            "emb_filename": self.emb_filename,
            "pos_tag_embedding_size": self.pos_tag_embedding_size,
            "use_bert": self.use_bert,
            "model_type": self.model_type,
            "save_type": self.save_type,
        }

    def save_model_to_file(self, fn):
        """
        Saves this model to file, also encodes class inits in the model's json
        """
        try:
            js = json.loads(self.model.to_json())
        except:
            logging.debug('keras to_json bug, only save hyperparams')
            js = {}

        # Add this model's params
        js["rnn"] = self.to_json()
        with open(fn, 'w') as fout:
            json.dump(js, fout)


    def sample_labels(self, y, num_of_sents = 5, num_of_samples = 10,
                      num_of_classes = 3, start_index = 5, get_prob = True):
        """
        Get a sense of how labels in y look like
        """
        classes = self.classes_()
        ret = []
        for sent in y[:num_of_sents]:
            cur = []
            for word in sent[start_index: start_index + num_of_samples]:
                sorted_prob = am(word)
                cur.append([(classes[ind], word[ind]) if get_prob else classes[ind]
                            for ind in sorted_prob[:num_of_classes]])
            ret.append(cur)
        return ret

class Sentence:
    """
    Prepare sentence sample for encoding
    """
    def __init__(words, pred_index):
        """
        words - list of strings representing the words in the sentence.
        pred_index - int representing the index of the current predicate for which to predict OIE extractions
        """

class Sample:
    """
    Single sample representation.
    Containter which names spans in the input vector to simplify access
    """
    def __init__(self, word):
        self.word = word

    def encode(self):
        """
        Encode this sample as vector as input for rnn,
        Probably just concatenating members in the right order.
        """
        return self.word

class Pad_sample(Sample):
    """
    A dummy sample used for padding
    """
    def __init__(self):
        Sample.__init__(self, word = 0)

def pad_sequences(sequences, pad_func, maxlen = None):
    """
    Similar to keras.preprocessing.sequence.pad_sequence but using Sample as higher level
    abstraction.
    pad_func is a pad class generator.
    """
    ret = []

    # Determine the maxlen
    max_value = max(map(len, sequences))
    if maxlen is None:
        maxlen = max_value

    # Pad / truncate (done this way to deal with np.array)
    for sequence in sequences:
        cur_seq = list(sequence[:maxlen])
        cur_seq.extend([pad_func()] * (maxlen - len(sequence)))
        ret.append(cur_seq)
    return ret



def load_pretrained_rnn(restore_dir):
    """ Static trained model loader function """
    # load model configuration
    rnn_params = json.load(open(os.path.join(restore_dir, "model.json")))["rnn"]

    # load model architectures and weights
    logging.info('loading model config from: {}'.format(restore_dir))
    rnn = RNN_model(restore_dir=restore_dir, **rnn_params) # model_dir not specified

    # compile model
    rnn.model_fn()

    return rnn


# Helper functions

## Argmaxes
am = lambda myList: [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse= True)]

if __name__ == "__main__":
    from pprint import pprint
    args = docopt(__doc__)
    #logging.debug(args)
    test_fn = args["--test"]

    if args["--train"] is not None:
        train_fn = args["--train"]
        dev_fn = args["--dev"]

        # load parameters
        if args["--load_hyperparams"] is not None:
            # load hyperparams from json file
            json_fn = args["--load_hyperparams"]
            logging.info('load model config from: {}'.format(json_fn))
            rnn_params = json.load(open(json_fn))["rnn"]
            #rnn_params["classes"] = None  # Just to make sure the model computes the correct labels
            if "classes" in rnn_params:
                logging.info('the order of classes matters! {}'.format(rnn_params['classes']))

        else:
            # Use some default params
            rnn_params = {"sent_maxlen":  20,
                          "hidden_units": pow(2, 10),
                          "num_of_latent_layers": 2,
                          "emb_filename": None,
                          "epochs": 10,
                          "trainable_emb": True,
                          "batch_size": 50,
                          "emb_filename": "../pretrained_word_embeddings/glove.6B.50d.txt",
                          "use_bert": False,
                          "model_type": "tag",
                          "save_type": "tag",
            }
        #logging.debug("hyperparams:\n{}".format(pformat(rnn_params)))

        # save and restore path
        if args["--saveto"] is not None:
            model_dir = os.path.join(args["--saveto"], "{}/".format(time.strftime("%d_%m_%Y_%H_%M")))
        else:
            model_dir = "../models/{}/".format(time.strftime("%d_%m_%Y_%H_%M"))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logging.info('save model to: {}'.format(model_dir))
        restore_dir = None
        if args['--restorefrom'] is not None:
            restore_dir = args['--restorefrom']
            logging.info('restore model from: {}'.format(restore_dir))

        rnn = RNN_model(model_dir=model_dir, restore_dir=restore_dir, **rnn_params)
        rnn.train(train_fn, dev_fn)

    elif args["--pretrained"] is not None:
        json_fn = args['--load_hyperparams']
        logging.info('load model config from: {}'.format(json_fn))
        rnn_params = json.load(open(json_fn))["rnn"]

        restore_dir = args['--pretrained']
        logging.info('restore model from: {}'.format(restore_dir))

        rnn = RNN_model(restore_dir=restore_dir, **rnn_params)
        rnn.test_confidence_model(*test_fn.split(':'))
        '''
        rnn = load_pretrained_rnn(args["--pretrained"])
        Y, y, metrics = rnn.test(test_fn,
                                 eval_metrics = [("F1 (micro)",
                                                  lambda Y, y: metrics.f1_score(Y, y,
                                                                                average = 'micro')),
                                                 ("Precision (micro)",
                                                  lambda Y, y: metrics.precision_score(Y, y,
                                                                                       average = 'micro')),
                                                 ("Recall (micro)",
                                                  lambda Y, y: metrics.recall_score(Y, y,
                                                                                    average = 'micro')),
                                                 ("Accuracy", metrics.accuracy_score),
                                             ])
        '''
"""
- the sentence max length is an important factor on convergence.
This makes sense, shorter sentences are easier to memorize.
The model was able to crack 20 words sentences pretty easily, but seems to be having a harder time with
40 words sentences.
Need to find a better balance.

- relu activation seems to have a lot of positive impact

- The batch size also seems to be having a lot of effect, but I'm not sure how to account for that.

- Maybe actually *increasing* dimensionalty would be better?
There are many ways to crack the train set - we want the model to be free in more
dimensions to allow for more flexibility while still fitting training data.

Ideas:

- test performance on arguments vs. adjuncts.
"""
