""" Usage:
    model [--train=TRAIN_FN] [--dev=DEV_FN] --test=TEST_FN [--pretrained=MODEL_DIR] [--load_hyperparams=MODEL_JSON] [--saveto=MODEL_DIR]
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) # append parent path

import numpy as np
import math
import pandas
import time
from docopt import docopt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, \
    TimeDistributed, merge, Bidirectional, Dropout
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

import json
import pdb
from keras.models import model_from_json
import logging
logging.basicConfig(level = logging.DEBUG)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    def __init__(self,  model_fn, sent_maxlen = None, emb_filename = None,
                 batch_size = 5, seed = 42, sep = '\t',
                 hidden_units = pow(2, 7),trainable_emb = True,
                 emb_dropout = 0.1, num_of_latent_layers = 2,
                 epochs = 10, pred_dropout = 0.1, model_dir = "./models/",
                 classes = None, pos_tag_embedding_size = 5,
                 use_bert = False,
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
        self.model_fn = lambda : model_fn(self)
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

        np.random.seed(self.seed)

    def get_callbacks(self, X):
        """
        Sets these callbacks as a class member.
        X is the encoded dataset used to print a sample of the output.
        Callbacks created:
        1. Sample output each epoch
        2. Save best performing model each epoch
        """

        sample_output_callback = LambdaCallback(on_epoch_end = lambda epoch, logs:\
                                                logging.debug(pformat(self.sample_labels(self.model.predict(X)))))
        checkpoint = ModelCheckpoint(os.path.join(self.model_dir,
                                                  "weights.hdf5"),
                                     verbose = 1,
                                     save_best_only = False)   # TODO: is there a way to save by best val_acc?

        #return [sample_output_callback,
        #        checkpoint]
        return [checkpoint]

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
        try:
            return self.encoder.classes_
        except:
            return self.classes

    def train_and_test(self, train_fn, test_fn):
        """
        Train and then test on given files
        """
        logging.info("Training..")
        self.train(train_fn)
        logging.info("Testing..")
        return self.test(test_fn)
        logging.info("Done!")

    def train(self, train_fn, dev_fn):
        """
        Train this model on a given train dataset
        Dev test is used for model checkpointing
        """
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
                       callbacks = self.get_callbacks(X_train))

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
        num_of_samples = np.ceil(float(len(sent)) / self.sent_maxlen) * self.sent_maxlen
        num_of_samples = int(num_of_samples)

        # Run RNN for each predicate on this sentence
        for ind, pred in preds:
            cur_sample = self.create_sample(sent, ind)
            X = self.encode_inputs([cur_sample])
            ret.append(((ind, pred),
                        [(self.consolidate_label(label), float(prob))
                         for (label, prob) in
                         self.transform_output_probs(self.model.predict(X),           # "flatten" and truncate
                                                     get_prob = True).reshape(num_of_samples,
                                                                              2)[:len(sent)]]))
        return ret

    def create_sample(self, sent, head_pred_id):
        """
        Return a dataframe which could be given to encode_inputs
        """
        return pandas.DataFrame({"word": sent,
                                 "run_id": [-1] * len(sent), # Mock running id
                                 "head_pred_id": head_pred_id})

    def test(self, test_fn, eval_metrics):
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

    def load_dataset(self, fn):
        """
        Load a supervised OIE dataset from file
        """
        df = pandas.read_csv(fn,
                             sep = self.sep,
                             header = 0,
                             keep_default_na = False)

        # Encode one-hot representation of the labels
        if self.classes_() is None:
            self.encoder.fit(df.label.values)

        # Split according to sentences and encode
        sents = self.get_sents_from_df(df)
        logging.info("{} has {} sentences".format(fn, len(sents)))
        return (self.encode_inputs(sents),
                self.encode_outputs(sents))

    def get_sents_from_df(self, df):
        """
        Split a data frame by rows accroding to the sentences
        """
        return [df[df.run_id == run_id]
                for run_id
                in sorted(set(df.run_id.values))]

    def get_fixed_size(self, sents):
        """
        Partition sents into lists of sent_maxlen elements
        (execept the last in each sentence, which might be shorter)
        """
        return [sent[s_ind : s_ind + self.sent_maxlen]
                for sent in sents
                for s_ind in range(0, len(sent), self.sent_maxlen)]

    def get_head_pred_word(self, full_sent):
        """
        Get the head predicate word from a full sentence conll.
        """
        assert(len(set(full_sent.head_pred_id.values)) == 1) # Sanity check
        pred_ind = full_sent.head_pred_id.values[0]

        return full_sent.word.values[pred_ind] \
            if pred_ind != -1 \
               else full_sent.pred.values[0].split(" ")[0]

    def encode_inputs(self, sents):
        """
        Given a dataframe which is already split to sentences,
        encode inputs for rnn classification.
        Should return a dictionary of sequences of sample of length maxlen.
        """
        word_inputs = []
        pred_inputs = []
        pos_inputs = []

        # Preproc to get all preds per run_id
        # Sanity check - make sure that all sents agree on run_id
        assert(all([len(set(sent.run_id.values)) == 1
                    for sent in sents]))
        run_id_to_pred = dict([(int(sent.run_id.values[0]),
                                self.get_head_pred_word(sent))
                               for sent in sents])

        # Construct a mapping from running word index to pos
        word_id_to_pos = {}
        for sent in sents:
            indices = sent.index.values
            words = sent.word.values

            for index, word in zip(indices,
                                   spacy_ws(" ".join(words))):
                word_id_to_pos[index] = word.tag_

        fixed_size_sents = self.get_fixed_size(sents)

        for sent in fixed_size_sents:

            assert(len(set(sent.run_id.values)) == 1)

            word_indices = sent.index.values
            sent_words = sent.word.values
            # must lowercase all the words before mapping to int because of
            # the special symbols in bert
            sent_words = np.array([w.lower() for w in sent_words])

            if self.use_bert:
                sent_words = np.insert(sent_words, 0, '[CLS]')
                sent_words = np.append(sent_words, '[SEP]')

            sent_str = " ".join(sent_words)

            pos_tags_encodings = [(SPACY_POS_TAGS.index(word_id_to_pos[word_ind]) \
                                   if word_id_to_pos[word_ind] in SPACY_POS_TAGS \
                                   else 0)
                                  for word_ind
                                  in word_indices]

            word_encodings = [self.emb.get_word_index(w, lower=False)
                              for w in sent_words]

            # Same pred word encodings for all words in the sentence
            pred_word = run_id_to_pred[int(sent.run_id.values[0])]
            pred_word_encodings = [self.emb.get_word_index(pred_word)
                                    for _ in sent_words]

            word_inputs.append([Sample(w) for w in word_encodings])
            pred_inputs.append([Sample(w) for w in pred_word_encodings])
            pos_inputs.append([Sample(pos) for pos in pos_tags_encodings])

        # Pad / truncate to desired maximum length
        ret = defaultdict(lambda: [])

        input_titles = ["word_inputs", "predicate_inputs", "postags_inputs"]
        input_tensors = [word_inputs, pred_inputs, pos_inputs]
        if self.use_bert:
            input_pad_lens = [self.sent_maxlen+2, self.sent_maxlen+2, self.sent_maxlen]
        else:
            input_pad_lens = [self.sent_maxlen, self.sent_maxlen, self.sent_maxlen]
        for name, sequence, pad_len in zip(input_titles, input_tensors, input_pad_lens):
            for samples in pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = pad_len):
                ret[name].append([sample.encode() for sample in samples])
        input_data = {k: np.array(v) for k, v in ret.iteritems()}
        if self.use_bert:
            # bert has more input
            input_data['word_segment_inputs'] = np.zeros_like(input_data['word_inputs'])
            input_data['predicate_segment_inputs'] = np.zeros_like(input_data['predicate_inputs'])
        return input_data

    def encode_outputs(self, sents):
        """
        Given a dataframe split to sentences, encode outputs for rnn classification.
        Should return a list sequence of sample of length maxlen.
        """
        output_encodings = []
        sents = self.get_fixed_size(sents)
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
        return [classes.index(label) for label in labels]

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

    def set_model_from_file(self):
        """
        Receives an instance of RNN and returns a model from the self.model_dir
        path which should contain a file named: model.json,
        and a single file with the hdf5 extension.
        Note: Use this function for a pretrained model, running model training
        on the loaded model will override the files in the model_dir
        """
        from glob import glob

        weights_fn = glob(os.path.join(self.model_dir, "*.hdf5"))
        assert len(weights_fn) == 1, "More/Less than one weights file in {}: {}".format(self.model_dir,
                                                                                        weights_fn)
        weights_fn = weights_fn[0]
        logging.debug("Weights file: {}".format(weights_fn))
        #self.model = model_from_json(open(os.path.join(self.model_dir,
        #                                               "./model.json")).read())
        self.set_vanilla_model(dump_json=False)

        self.model.load_weights(weights_fn)
        self.model.compile(optimizer="adam",
                           loss='categorical_crossentropy',
                           metrics = ["accuracy"])

    def set_vanilla_model(self, dump_json=True):
        """
        Set a Keras model for predicting OIE as a member of this class
        Can be passed as model_fn to the constructor
        """
        logging.debug("Setting vanilla model")
        # Build model

        ## Embedding Layer
        word_embedding_layer = self.embed_word()
        pos_embedding_layer = self.embed_pos()

        ## Deep layers
        latent_layers = self.stack_latent_layers(self.num_of_latent_layers)

        ## Dropout
        dropout = Dropout(self.pred_dropout)

        ## Prediction
        predict_layer = self.predict_classes()

        ## Prepare input features, and indicate how to embed them
        if self.use_bert:
            # bert takes two tensors as input: tokens and segment
            inputs_and_embeddings = \
                [([Input(shape=(self.sent_maxlen+2,), dtype="int32", name="word_inputs"),
                   Input(shape=(self.sent_maxlen+2,), dtype="int32", name="word_segment_inputs")],
                word_embedding_layer),
                ([Input(shape=(self.sent_maxlen+2,), dtype="int32", name="predicate_inputs"),
                   Input(shape=(self.sent_maxlen+2,), dtype="int32", name="predicate_segment_inputs")],
                word_embedding_layer)]
        else:
            inputs_and_embeddings = \
                [(Input(shape=(self.sent_maxlen,), dtype="int32", name="word_inputs"),
                word_embedding_layer),
                (Input(shape=(self.sent_maxlen,), dtype="int32", name="predicate_inputs"), 
                word_embedding_layer)]
        inputs_and_embeddings.append(
            (Input(shape=(self.sent_maxlen,), dtype="int32", name = "postags_inputs"),
            pos_embedding_layer))


        ## Concat all inputs and run on deep network
        #output = predict_layer(dropout(latent_layers(merge([embed(inp)
        #                                                    for inp, embed in inputs_and_embeddings],
        #                                                   mode = "concat",
        #                                                   concat_axis = -1))))
        output = predict_layer(dropout(latent_layers(concatenate([embed(inp)
                                                            for inp, embed in inputs_and_embeddings],
                                                           axis = -1))))

        # Build model
        model_input = []
        for inps, embed in inputs_and_embeddings:
            if type(inps) is not list:
                inps = [inps]
            for inp in inps:
                model_input.append(inp)
        self.model = Model(input = model_input,
                           output = [output])

        # Loss
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
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



def load_pretrained_rnn(model_dir):
    """ Static trained model loader function """
    rnn_params = json.load(open(os.path.join(model_dir, "model.json")))["rnn"]

    logging.info("Loading model from: {}".format(model_dir))
    rnn = RNN_model(model_fn = RNN_model.set_model_from_file,
                    model_dir = model_dir,
                    **rnn_params)

    # Compile model
    rnn.model_fn()

    return rnn


# Helper functions

## Argmaxes
am = lambda myList: [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse= True)]

if __name__ == "__main__":
    from pprint import pprint
    args = docopt(__doc__)
    logging.debug(args)
    test_fn = args["--test"]

    if args["--train"] is not None:
        train_fn = args["--train"]
        dev_fn = args["--dev"]

        if args["--load_hyperparams"] is not None:
            # load hyperparams from json file
            json_fn = args["--load_hyperparams"]
            logging.info("Loading model from: {}".format(json_fn))
            rnn_params = json.load(open(json_fn))["rnn"]
            rnn_params["classes"] = None  # Just to make sure the model computes the correct labels

        else:
            # Use some default params
            rnn_params = {"sent_maxlen":  20,
                          "hidden_units": pow(2, 10),
                          "num_of_latent_layers": 2,
                          "emb_filename": emb_filename,
                          "epochs": 10,
                          "trainable_emb": True,
                          "batch_size": 50,
                          "emb_filename": "../pretrained_word_embeddings/glove.6B.50d.txt",
                          "use_bert": False,
            }


        logging.debug("hyperparams:\n{}".format(pformat(rnn_params)))
        if args["--saveto"] is not None:
            model_dir = os.path.join(args["--saveto"], "{}/".format(time.strftime("%d_%m_%Y_%H_%M")))
        else:
            model_dir = "../models/{}/".format(time.strftime("%d_%m_%Y_%H_%M"))
        logging.debug("Saving models to: {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        rnn = RNN_model(model_fn = RNN_model.set_vanilla_model,
                        model_dir = model_dir,
                        **rnn_params)
        rnn.train(train_fn, dev_fn)

    elif args["--pretrained"] is not None:
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
