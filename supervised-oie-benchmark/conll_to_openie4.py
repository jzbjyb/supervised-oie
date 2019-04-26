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

def pred_arg_to_eval(sub_tokens, tokens):
    tokens = [t[0] for t in sub_tokens]
    inds = [t[1] for t in sub_tokens]
    return str((' '.join(tokens), inds))

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
    parser.add_argument('--inp', type=str, help='input conll files separated by')
    parser.add_argument('--out', type=str, help='output file in openie4 format')
    parser.add_argument('--rm_coor', help='whether to remove predicates with multiple extractions', 
        action='store_true')
    parser.add_argument('--format', help='output format', 
        choices=['openie4', 'eval'], default='openie4')
    opt = parser.parse_args()

    most_n_args = 6
    SEP = '|;|;|'

    df = pd.read_csv(opt.inp, sep='\t', header=0, keep_default_na=False, quoting=3)
    sents = Extraction.get_sents_from_df(df)
    useless_n_sam = 0
    n_sam = 0
    all_sent_pred_set, rm_set = set(), set()
    results = []
    with open(opt.out, 'w') as fout:
        for sent in sents:
            n_sam += 1
            words = sent.word.values
            tags = sent.label.values
            sent = ' '.join(words)
            pred = []
            args = [[] for i in range(most_n_args)] # collect at most most_n_args arguments
            args_touch = [False] * most_n_args
            last_ind = -1 # -1 for start and O, -2 for predicate, others for arguments
            try:
                for i, w, t in zip(range(len(words)), words, tags):
                    if t.startswith('P'):
                        # predicate
                        if last_ind != -2:
                            pred.append([])
                        pred[-1].append((w, i))
                        last_ind = -2
                    elif t.startswith('A'):
                        # we don't care about B and I here to avoid bugs from the conll file.
                        ai = int(t[1:t.find('-')])
                        if ai >= len(args):
                            print(t)
                            raise ValueError('get an extraction with more than {} args'.format(most_n_args))
                        if args_touch[ai] and last_ind != ai:
                            raise ConllNotValidError('not contiguous span')
                        args_touch[ai] = True
                        args[ai].append((w, i))
                        last_ind = ai
                    else:
                        last_ind = -1
            except ConllNotValidError:
                useless_n_sam += 1
                continue
            if len(pred) != 1:
                print('predicate span is not unique')
                useless_n_sam += 1
                continue
            pred = pred[0]
            if len(args) <= 0 or len(pred) <= 0:
                useless_n_sam += 1
                continue
            # remove extractions with the same predicate
            sent_pred_hash = '{}<rm_coor>{}'.format(sent, [i[1] for i in pred])
            if opt.rm_coor and sent_pred_hash in all_sent_pred_set:
                useless_n_sam += 1
                if sent_pred_hash not in rm_set:
                    useless_n_sam += 1
                    rm_set.add(sent_pred_hash)
            else:
                all_sent_pred_set.add(sent_pred_hash)
            if opt.format == 'openie4':
                pred_str = pred_to_openie4(pred, words)
                args_str = [arg_to_openie4(arg, words) for arg in args if len(arg) > 0]
                for arg in args_str[1:]:
                    if arg.find(SEP) >= 0:
                        raise ValueError('openie4 format conflict')
                results.append((sent_pred_hash, '{}\t\t{}\t{}\t{}\t{}\n'.format(
                    0, args_str[0], pred_str, SEP.join(args_str[1:]), sent)))
            elif opt.format == 'eval':
                pred_str = pred_arg_to_eval(pred, words)
                args_str = [pred_arg_to_eval(arg, words) for arg in args if len(arg) > 0]
                results.append((sent_pred_hash, '{}\t{}\t{}\n'.format(
                    sent, pred_str, '\t'.join(args_str))))
        # write to file
        for r in results:
            if r[0] in rm_set:
                continue
            fout.write(r[1])
    print('totally {} useless samples'.format(useless_n_sam))
    print('from {} to {} samples'.format(n_sam, n_sam-useless_n_sam))
