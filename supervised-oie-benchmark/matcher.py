import string
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
from nltk.parse import CoreNLPParser
import nltk
import numpy as np
import time
from oie_readers.extraction import Extraction
from difflib import SequenceMatcher

parser = CoreNLPParser(url='http://localhost:9001')

class Matcher:
    @staticmethod
    def traverse_tree(tree, depth, depth_list, debug=False):
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                Matcher.traverse_tree(subtree, depth + 1, depth_list, debug=debug)
            elif type(subtree) == unicode:
                if debug:
                    print(subtree)
                for word in subtree.split(u'\xa0'): # to avoid stanford nlp bug
                    depth_list.append(depth)
            else:
                print(type(subtree))

    @staticmethod
    def stanford_parse(sentence):
        pt = list(parser.parse(sentence))
        dl = []
        Matcher.traverse_tree(pt, 0, dl)
        if len(dl) < len(sentence):
            # TODO: there is a mismatching between parser output and raw sentence
            print('>> --- stanfordnlp bug report ---')
            print(sentence)
            print(list(parser.parse(sentence)))
            dl += (len(sentence) - len(dl)) * (dl[-1:] if len(dl) > 0 else [0])
            print('<< --- stanfordnlp bug report ---')
            assert len(dl) == len(sentence)
        elif len(dl) > len(sentence):
            raise Exception('parser longer bug')
        return dl

    @staticmethod
    def get_syntactic_head(sentence, start_ind, end_ind, depth_list=None):
        if depth_list is None:
            depth_list = Matcher.stanford_parse(sentence)
        return sentence[np.argmin(depth_list[start_ind:end_ind]) + start_ind]

    @staticmethod
    def arg_span_based_cound(ref, ex):
        mla = min(len(ref.args), len(ex.args))
        tp, fp, fn = 0, 0, 0
        for i in range(mla):
            ref_arg = ref.args[i]
            arg = ex.args[i]
            # use position or str to evaluate
            if type(ref_arg) is tuple and type(arg) is tuple:
                matched = Extraction.position_equal(ref_arg[1], arg[1])
            else:
                matched = ex.elementToStr(arg, print_indices=False) == \
                          ref.elementToStr(ref_arg, print_indices=False)
            if matched:
                tp += 1
            else:
                fp += 1
                fn += 1
        fp += len(ex.args) - mla
        fn += len(ref.args) - mla
        return tp, fp, fn

    @staticmethod
    def bowMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        A binary function testing for exact lexical match (ignoring ordering) between reference
        and predicted extraction
        """
        s1 = ref.bow()
        s2 = ex.bow()
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()

        s1Words = s1.split(' ')
        s2Words = s2.split(' ')

        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)

        return sorted(s1Words) == sorted(s2Words)

    @staticmethod
    def predMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the predicate
        """
        s1 = ref.elementToStr(ref.pred, print_indices=False)
        s2 = ex.elementToStr(ex.pred, print_indices=False)
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()

        s1Words = s1.split(' ')
        s2Words = s2.split(' ')

        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)

        return s1Words  == s2Words, int(s1Words  == s2Words)

    @staticmethod
    def predLexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the predicate
        """
        sRef = ref.elementToStr(ref.pred, print_indices=False).split(' ')
        sEx = ex.elementToStr(ex.pred, print_indices=False).split(' ')

        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)
        return coverage > Matcher.LEXICAL_THRESHOLD, coverage

    @staticmethod
    def predHeadMatch(ref, ex, ignoreStopwords, ignoreCase):
        if type(ref.heads[0]) is tuple and type(ex.pred) is tuple:
            # use position
            cont = Extraction.position_contain(ref.heads[0][1], ex.pred[1])
            return cont, int(cont)
        else:
            # use str
            pred = ' ' + ex.elementToStr(ex.pred, print_indices=False) + ' '
            pred = pred.find(' ' + ref.elementToStr(ref.heads[0], print_indices=False) + ' ')
            return pred >= 0, int(pred >= 0)

    @staticmethod
    def predHeadLexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        mb, ms = Matcher.predHeadMatch(ref, ex, ignoreStopwords, ignoreCase)
        if not mb:
            return mb, ms
        ep = ex.elementToStr(ex.pred, print_indices=False)
        rp = ref.elementToStr(ref.pred, print_indices=False)
        s = SequenceMatcher(None, ep, rp).ratio()
        return True, s

    @staticmethod
    def argLexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the arguments
        """
        sRef = ' '.join([ref.elementToStr(elem, print_indices=False) for elem in ref.args]).split(' ')
        sEx = ' '.join([ex.elementToStr(elem, print_indices=False) for elem in ex.args]).split(' ')

        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)


        return coverage > Matcher.LEXICAL_THRESHOLD, coverage

    @staticmethod
    def argHeadMatch(ref, ex, ignoreStopwords, ignoreCase):
        if len(ref.args) != len(ex.args):
            return False, 0
        for i, arg in enumerate(ex.args):
            head = ref.heads[i + 1]
            if type(head) is tuple and type(arg) is tuple:
                # use position to evaluate
                if not Extraction.position_contain(head[1], arg[1]):
                    return False, 0
            else:
                # use str to evaluate
                arg = ' ' + ex.elementToStr(arg, print_indices=False) + ' '
                ind = arg.find(' ' + ref.elementToStr(head, print_indices=False) + ' ')
                if ind < 0:
                    return False, 0
        return True, 1

    @staticmethod
    def argHeadLexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        mb, ms = Matcher.argHeadMatch(ref, ex, ignoreStopwords, ignoreCase)
        if not mb:
            return mb, ms
        s_li = []
        for i, arg in enumerate(ex.args):
            ref_arg = ref.args[i]
            ea = ex.elementToStr(arg, print_indices=False)
            ra = ref.elementToStr(ref_arg, print_indices=False)
            s_li.append(SequenceMatcher(None, ea, ra).ratio())
        return True, np.mean(s_li)
    
    @staticmethod
    def argHeadMatchRelex(ref, ex, ignoreStopwords, ignoreCase):
        for head in ref.heads[1:]:
            matched = False
            for i, arg in enumerate(ex.args):
                if type(head) is tuple and type(arg) is tuple:
                    # use position to evaluate
                    if Extraction.position_contain(head[1], arg[1]):
                        matched = True
                        break
                else:
                    # use str to evaluate
                    arg = ' ' + ex.elementToStr(arg, print_indices=False) + ' '
                    ind = arg.find(' ' + ref.elementToStr(head, print_indices=False) + ' ')
                    if ind >= 0:
                        matched = True
                        break
            if not matched:
                return False, 0
        return True, 1

    @staticmethod
    def argHeadMatchExclude(ref, ex, ignoreStopwords, ignoreCase):
        if len(ref.args) > len(ex.args):
            return False, 0
        for i, head in enumerate(ref.heads[1:]):
            arg = ex.args[i]
            if type(head) is tuple and type(arg) is tuple:
                # use position to evaluate
                if not Extraction.position_contain(head[1], arg[1]):
                    return False, 0
            else:
                # use str to evaluate
                arg = ' ' + ex.elementToStr(arg, print_indices=False) + ' '
                ind = arg.find(' ' + ref.elementToStr(head, print_indices=False) + ' ')
                if ind < 0:
                    return False, 0
        return True, 1

    @staticmethod
    def argHeadLexicalMatchExclude(ref, ex, ignoreStopwords, ignoreCase):
        mb, ms = Matcher.argHeadMatchExclude(ref, ex, ignoreStopwords, ignoreCase)
        if not mb:
            return mb, ms
        s_li = []
        for i, _ in enumerate(ref.heads[1:]):
            arg = ex.args[i]
            ref_arg = ref.args[i]
            ea = ex.elementToStr(arg, print_indices=False)
            ra = ref.elementToStr(ref_arg, print_indices=False)
            s_li.append(SequenceMatcher(None, ea, ra).ratio())
        return True, np.mean(s_li)

    @staticmethod
    def bleuMatch(ref, ex, ignoreStopwords, ignoreCase):
        sRef = ref.bow()
        sEx = ex.bow()
        bleu = sentence_bleu(references = [sRef.split(' ')], hypothesis = sEx.split(' '))
        return bleu > Matcher.BLEU_THRESHOLD

    @staticmethod
    def lexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        sRef = ref.bow().split(' ')
        sEx = ex.bow().split(' ')
        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)


        return coverage > Matcher.LEXICAL_THRESHOLD, coverage

    @staticmethod
    def exactlySameMatch(ref, ex, ignoreStopwords, ignoreCase):
        if len(ref.args) != len(ex.args):
            return False, 0
        for i, ref_arg in enumerate(ref.args):
            if ref.elementToStr(ref_arg, print_indices=False) != ex.elementToStr(ex.args[i], print_indices=False):
                return False, 0
        if ref.elementToStr(ref.pred, print_indices=False) != ex.elementToStr(ex.pred, print_indices=False):
            return False, 0
        return True, 1

    @staticmethod
    def predArgLexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        pred, pred_score = Matcher.predLexicalMatch(ref, ex, ignoreStopwords, ignoreCase)
        arg, arg_score = Matcher.argLexicalMatch(ref, ex, ignoreStopwords, ignoreCase)
        return pred and arg, min(pred_score, arg_score)

    @staticmethod
    def predArgHeadMatch(ref, ex, ignoreStopwords, ignoreCase):
        pred, pred_score = Matcher.predHeadMatch(ref, ex, ignoreStopwords, ignoreCase)
        arg, arg_score = Matcher.argHeadMatch(ref, ex, ignoreStopwords, ignoreCase)
        return pred and arg, min(pred_score, arg_score)

    @staticmethod
    def predArgHeadLexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        pred, pred_score = Matcher.predHeadLexicalMatch(ref, ex, ignoreStopwords, ignoreCase)
        arg, arg_score = Matcher.argHeadLexicalMatch(ref, ex, ignoreStopwords, ignoreCase)
        return pred and arg, np.mean([pred_score, arg_score])

    @staticmethod
    def predArgHeadMatchRelex(ref, ex, ignoreStopwords, ignoreCase):
        pred, pred_score = Matcher.predHeadMatch(ref, ex, ignoreStopwords, ignoreCase)
        arg, arg_score = Matcher.argHeadMatchRelex(ref, ex, ignoreStopwords, ignoreCase)
        return pred and arg, min(pred_score, arg_score)

    @staticmethod
    def predArgHeadMatchExclude(ref, ex, ignoreStopwords, ignoreCase):
        pred, pred_score = Matcher.predHeadMatch(ref, ex, ignoreStopwords, ignoreCase)
        arg, arg_score = Matcher.argHeadMatchExclude(ref, ex, ignoreStopwords, ignoreCase)
        return pred and arg, min(pred_score, arg_score)

    @staticmethod
    def predArgHeadLexicalMatchExclude(ref, ex, ignoreStopwords, ignoreCase):
        pred, pred_score = Matcher.predHeadLexicalMatch(ref, ex, ignoreStopwords, ignoreCase)
        arg, arg_score = Matcher.argHeadLexicalMatchExclude(ref, ex, ignoreStopwords, ignoreCase)
        return pred and arg, np.mean([pred_score, arg_score])

    @staticmethod
    def removeStopwords(ls):
        return [w for w in ls if w.lower() not in Matcher.stopwords]

    # CONSTANTS
    BLEU_THRESHOLD = 0.4
    LEXICAL_THRESHOLD = 0.5 # Note: changing this value didn't change the ordering of the tested systems
    stopwords = stopwords.words('english') + list(string.punctuation)





