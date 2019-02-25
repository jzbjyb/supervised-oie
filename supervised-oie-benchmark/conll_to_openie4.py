import argparse
import pandas as pd
from operator import itemgetter
from oie_readers.extraction import Extraction

class ConllNotValidError(Exception):
    pass

def len_in_char(text, encoding='utf8'):
    if type(text) is str:
        text = text.decode(encoding)
    if type(text) is not unicode:
        raise ValueError
    return len(text)

def arg_to_openie4(sub_tokens, tokens):
    st = len_in_char(' '.join(tokens[:sub_tokens[0][1]])) + (sub_tokens[0][1] > 0)
    sub = ' '.join(map(itemgetter(0), sub_tokens))
    ed = st + len_in_char(sub)
    return '{}({},List([{}, {})))'.format('SimpleArgument', sub, st, ed)

def pred_to_openie4(sub_tokens, tokens):
    sts, eds = [], []
    for w in sub_tokens:
        st = len_in_char(' '.join(tokens[:w[1]])) + (w[1] > 0)
        ed = st + len_in_char(w[0])
        sts.append(st)
        eds.append(ed)
    return '{}({},List({}))'.format(
        'Relation', ' '.join(map(itemgetter(0), sub_tokens)),
        ', '.join('[{}, {})'.format(st, ed) for st, ed in zip(sts, eds)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert conll file (used in training) to openie4 format')
    parser.add_argument('-inp', type=str, help='input conll files separated by')
    parser.add_argument('-out', type=str, help='output file in openie4 format')
    opt = parser.parse_args()

    most_n_args = 6
    SEP = '|;|;|'

    df = pd.read_csv(opt.inp, sep='\t', header=0, keep_default_na=False, quoting=3)
    sents = Extraction.get_sents_from_df(df)
    useless_n_sam = 0
    n_sam, final_n_sam = 0, 0
    with open(opt.out, 'w') as fout:
        for sent in sents:
            n_sam += 1
            words = sent.word.values
            tags = sent.label.values
            sent = ' '.join(words)
            pred = []
            args = [[] for i in range(most_n_args)] # collect at most most_n_args arguments
            args_touch = [False] * most_n_args
            last_arg_ind = -1
            try:
                for i, w, t in zip(range(len(words)), words, tags):
                    if t.startswith('P'):
                        # predicate
                        pred.append((w, i))
                        last_arg_ind = -1
                    elif t.startswith('A'):
                        # we don't care about B and I here to avoid bugs from the conll file.
                        ai = int(t[1:t.find('-')])
                        if ai >= len(args):
                            print(t)
                            raise ValueError('get an extraction with more than {} args'.format(most_n_args))
                        if args_touch[ai] and last_arg_ind != ai:
                            raise ConllNotValidError('not contiguous span')
                        args_touch[ai] = True
                        args[ai].append((w, i))
                        last_arg_ind = ai
                    else:
                        last_arg_ind = -1
            except ConllNotValidError:
                useless_n_sam += 1
                continue
            pred_str = pred_to_openie4(pred, words)
            args_str = [arg_to_openie4(arg, words) for arg in args if len(arg) > 0]
            if len(args_str) <= 0 or len(pred) <= 0:
                useless_n_sam += 1
                continue
            for arg in args_str[1:]:
                if arg.find(SEP) >= 0:
                    raise ValueError('openie4 format conflict')
            final_n_sam += 1
            fout.write('{}\t\t{}\t{}\t{}\t{}\n'.format(
                0, args_str[0], pred_str, SEP.join(args_str[1:]), sent))
    print('totally {} useless samples'.format(useless_n_sam))
    print('from {} to {} samples'.format(n_sam, final_n_sam))
