'''
Usage:
   benchmark --gold=GOLD_OIE --out=OUTPUT_FILE (--stanford=STANFORD_OIE | --ollie=OLLIE_OIE |--reverb=REVERB_OIE | --clausie=CLAUSIE_OIE | --openiefour=OPENIEFOUR_OIE | --props=PROPS_OIE | --tabbed=TABBED_OIE) [--exactMatch | --predMatch | --argMatch] [--error-file=ERROR_FILE]

Options:
  --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
  --out-OUTPUT_FILE            The output file, into which the precision recall curve will be written.
  --clausie=CLAUSIE_OIE        Read ClausIE format from file CLAUSIE_OIE.
  --ollie=OLLIE_OIE            Read OLLIE format from file OLLIE_OIE.
  --openiefour=OPENIEFOUR_OIE  Read Open IE 4 format from file OPENIEFOUR_OIE.
  --props=PROPS_OIE            Read PropS format from file PROPS_OIE
  --reverb=REVERB_OIE          Read ReVerb format from file REVERB_OIE
  --stanford=STANFORD_OIE      Read Stanford format from file STANFORD_OIE
  --tabbed=TABBED_OIE          Read simple tab format file, where each line consists of:
                                sent, prob, pred,arg1, arg2, ...
  --exactmatch                 Use exact match when judging whether an extraction is correct.
'''
import docopt
import string
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import re
import logging
import pdb
from itertools import groupby
from termcolor import colored
from xtermcolor import colorize
import random
logging.basicConfig(level = logging.INFO)

from oie_readers.stanfordReader import StanfordReader
from oie_readers.ollieReader import OllieReader
from oie_readers.reVerbReader import ReVerbReader
from oie_readers.clausieReader import ClausieReader
from oie_readers.openieFourReader import OpenieFourReader
from oie_readers.propsReader import PropSReader
from oie_readers.tabReader import TabReader

from oie_readers.goldReader import GoldReader
from matcher import Matcher
from operator import itemgetter

SEED = 2019
random.seed(SEED)
np.random.seed(SEED)

class ColoredExtraction(object):
    def __init__(self, extraction):
        self.extraction = extraction

    @staticmethod
    def color_show():
        print(colorize('predicate color', ansi=196))
        for i in range(5):
            print(colorize('arg{} color'.format(i), ansi=40 + (i * 1)))

    @staticmethod
    def gen_binary(starts, ends, length):
        pairs = list(zip(starts, ends))
        result = []
        for i in range(length):
            find = False
            for s,e in pairs:
                if i >= s and i < e:
                    find = True
                    break
            result.append(find)
        return result

    def __str__(self):
        pred = self.extraction.elementToStr(self.extraction.pred)
        args = [self.extraction.elementToStr(arg) for arg in self.extraction.args]
        sent = ' {} '.format(self.extraction.sent) # add space to avoid 'on' match 'money'
        pred_start = sent.find(' {} '.format(pred)) + 1
        pred_end = pred_start + len(pred)
        args_start = [sent.find(' {} '.format(arg)) + 1 for arg in args]
        args_end = [args_start[i] + len(arg) for i, arg in enumerate(args)]
        pred_cond = ColoredExtraction.gen_binary([pred_start], [pred_end], len(sent))
        args_cond = [ColoredExtraction.gen_binary([arg_start], [arg_end], len(sent)) \
            for arg_start, arg_end in zip(args_start, args_end)]
        result = []
        for i, ch in enumerate(sent):
            if pred_cond[i]:
                #ch = colored(ch, 'red')
                ch = colorize(ch, ansi=196)
            else:
                for argi, arg_cond in enumerate(args_cond):
                    if arg_cond[i]:
                        #ch = colored(ch, 'cyan')
                        ch = colorize(ch, ansi=40 + argi)
                        break
            result.append(ch)
        return ''.join(result)[1:-1]

class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''
    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        gr = GoldReader()
        gr.read(gold_fn)
        self.gold = gr.oie

    @staticmethod
    def get_n_random_sample_out_of(samples, n):
        return samples[np.random.permutation(len(samples))[:n]]

    def get_ana_banner(self, ind):
        result = [
            '# ====================== #',
            '# ===== ANALYSIS {} ===== #'.format(ind),
            '# ====================== #']
        return '\n'.join(result)

    def error_ana_bi(self, predicted1, preddicted2):
        pass

    def error_ana_uni(self, predicted, showcase=5):
        ColoredExtraction.color_show()
        '''
        ana1
        '''
        print(self.get_ana_banner(1))
        sent_empty = np.array([k for k in self.gold if k not in predicted])
        print('{} out of {} sentence have no extraction'.format(len(sent_empty), len(self.gold)))
        cases = Benchmark.get_n_random_sample_out_of(sent_empty, showcase)
        print('{} samples of them'.format(len(cases)))
        for s in cases:
            print('* {}'.format(s))
            for e in self.gold[s]:
                #print('\t{}'.format(e))
                print('\t{}'.format(ColoredExtraction(e)))
        '''
        ana2 
        '''
        print(self.get_ana_banner(2))
        wrong_ext = np.array([(s, e) for s in predicted for e in predicted[s] 
            if len(e.matched) == 0])
        num_ext_all = len([1 for s in predicted for e in predicted[s]])
        num_ext_wrong = len(wrong_ext)
        print('{} out of {} extractions are wrong'.format(num_ext_wrong, num_ext_all))
        wrong_ext_dict = dict((k, list(e[1] for e in g)) 
            for k, g in groupby(wrong_ext, lambda x: x[0]))
        cases = Benchmark.get_n_random_sample_out_of(
            np.array(list(wrong_ext_dict.items())), showcase)
        print('{} samples of them'.format(len(cases)))
        for s, exts in cases:
            print('* {}'.format(s))
            print('\t--- ground truth ---')
            for e in self.gold[s]:
                #print('\t{}'.format(e))
                print('\t{}'.format(ColoredExtraction(e)))
            print('\t--- wrong ---')
            for e in exts:
                #print('\t{}'.format(e))
                print('\t{}'.format(ColoredExtraction(e)))
        '''
        ana3
        '''
        print(self.get_ana_banner(3))
        correct_ext = np.array([(s, e) for s in predicted for e in predicted[s] 
            if len(e.matched) > 0])
        correct_ext_dict = dict((k, list(e[1] for e in g))
            for k, g in groupby(correct_ext, lambda x: x[0]))
        cases = Benchmark.get_n_random_sample_out_of(
            np.array(list(correct_ext_dict.items())), showcase)
        print('{} samples of correct'.format(len(cases)))
        for s, exts in cases:
            print('* {}'.format(s))
            print('\t--- ground truth ---')
            for e in self.gold[s]:
                #print('\t{}'.format(e))
                print('\t{}'.format(ColoredExtraction(e)))
            print('\t--- correct ---')
            for e in exts:
                #print('\t{}'.format(e))
                print('\t{}'.format(ColoredExtraction(e)))

    def compare(self, predicted, matchingFunc, output_fn, error_file = None):
        ''' Compare gold against predicted using a specified matching function.
            Outputs PR curve to output_fn '''

        y_true = []
        y_scores = []
        errors = []

        correctTotal = 0
        unmatchedCount = 0

        # It seems that we don't need normalize
        #predicted = Benchmark.normalizeDict(predicted)
        #gold = Benchmark.normalizeDict(self.gold)
        gold = self.gold

        for sent, goldExtractions in gold.items():
            if sent not in predicted:
                # The extractor didn't find any extractions for this sentence
                for goldEx in goldExtractions:
                    unmatchedCount += len(goldExtractions)
                    correctTotal += len(goldExtractions)
                continue

            predictedExtractions = predicted[sent]

            for goldEx in goldExtractions:
                correctTotal += 1
                found = False

                for predictedEx in predictedExtractions:
                    if output_fn in predictedEx.matched:
                        # This predicted extraction was already matched against a gold extraction
                        # Don't allow to match it again
                        continue

                    if matchingFunc(goldEx,
                                    predictedEx,
                                    ignoreStopwords = True,
                                    ignoreCase = True):

                        y_true.append(1)
                        y_scores.append(predictedEx.confidence)
                        predictedEx.matched.append(output_fn)
                        found = True
                        break

                if not found:
                    errors.append(goldEx.index)
                    unmatchedCount += 1

            for predictedEx in [x for x in predictedExtractions if (output_fn not in x.matched)]:
                # Add false positives
                y_true.append(0)
                y_scores.append(predictedEx.confidence)

        y_true = y_true
        y_scores = y_scores

        # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # to get to true recall we do:
        # r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        (p, r), optimal = Benchmark.prCurve(np.array(y_true), np.array(y_scores),
                                            recallMultiplier = ((correctTotal - unmatchedCount)/float(correctTotal)))
        logging.info("AUC: {}\n Optimal (precision, recall, F1, threshold): {}".format(auc(r, p),
                                                                                       optimal))

        # Write error log to file
        if error_file:
            logging.info("Writing {} error indices to {}".format(len(errors),
                                                                 error_file))
            with open(error_file, 'w') as fout:
                fout.write('\n'.join([str(error)
                                     for error
                                      in errors]) + '\n')

        # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            for cur_p, cur_r in sorted(zip(p, r), key = lambda (cur_p, cur_r): cur_r):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))

        return predicted

    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier):
        # Recall multiplier - accounts for the percentage examples unreached
        # Return (precision [list], recall[list]), (Optimal F1, Optimal threshold)
        y_scores = [score \
                    if not (np.isnan(score) or (not np.isfinite(score))) \
                    else 0
                    for score in y_scores]
        
        precision_ls, recall_ls, thresholds = precision_recall_curve(y_true, y_scores)
        recall_ls = recall_ls * recallMultiplier
        optimal = max([(precision, recall, f_beta(precision, recall, beta = 1), threshold)
                       for ((precision, recall), threshold)
                       in zip(zip(precision_ls[:-1], recall_ls[:-1]),
                              thresholds)],
                      key = itemgetter(2))  # Sort by f1 score

        return ((precision_ls, recall_ls),
                optimal)

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k), v) for k, v in d.items()])

    @staticmethod
    def normalizeKey(k):
        return Benchmark.removePunct(unicode(Benchmark.PTB_unescape(k.replace(' ','')), errors = 'ignore'))

    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s

    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s

    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)

    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Penn treebank bracket escapes
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'),]


def f_beta(precision, recall, beta = 1):
    """
    Get F_beta score from precision and recall.
    """
    beta = float(beta) # Make sure that results are in float
    return (1 + pow(beta, 2)) * (precision * recall) / ((pow(beta, 2) * precision) + recall)


f1 = lambda precision, recall: f_beta(precision, recall, beta = 1)




if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    logging.debug(args)

    if args['--stanford']:
        predicted = StanfordReader()
        predicted.read(args['--stanford'])

    if args['--props']:
        predicted = PropSReader()
        predicted.read(args['--props'])

    if args['--ollie']:
        predicted = OllieReader()
        predicted.read(args['--ollie'])

    if args['--reverb']:
        predicted = ReVerbReader()
        predicted.read(args['--reverb'])

    if args['--clausie']:
        predicted = ClausieReader()
        predicted.read(args['--clausie'])

    if args['--openiefour']:
        predicted = OpenieFourReader()
        predicted.read(args['--openiefour'])

    if args['--tabbed']:
        predicted = TabReader()
        predicted.read(args['--tabbed'])

    if args['--exactMatch']:
        matchingFunc = Matcher.argMatch

    elif args['--predMatch']:
        matchingFunc = Matcher.predMatch

    elif args['--argMatch']:
        matchingFunc = Matcher.argMatch

    else:
        matchingFunc = Matcher.lexicalMatch

    b = Benchmark(args['--gold'])
    out_filename = args['--out']

    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    compared_predicated = b.compare(predicted = predicted.oie,
              matchingFunc = matchingFunc,
              output_fn = out_filename,
              error_file = args["--error-file"])
    b.error_ana_uni(compared_predicated, showcase=5)

