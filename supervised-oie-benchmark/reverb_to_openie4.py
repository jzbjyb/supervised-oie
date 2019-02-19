import argparse
import pandas as pd
from operator import itemgetter
from oie_readers.extraction import Extraction
from conll_to_openie4 import arg_to_openie4, pred_to_openie4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert reverb file to openie4 format')
    parser.add_argument('-inp', type=str, help='input reverb files separated')
    parser.add_argument('-out', type=str, help='output file in openie4 format')
    opt = parser.parse_args()

    useless_n_sam = 0

    with open(opt.inp, 'r') as fin, open(opt.out, 'w') as fout:
        for l in fin:
            l = l.strip()
            if l == '':
                continue
            f, sid, arg0, rel, arg1, arg0s, arg0e, \
            rels, rele, arg1s, arg1e, conf, sent, = l.split('\t')[:13]
            arg0s, arg0e, rels, rele, arg1s, arg1e = list(map(int, l.split('\t')[5:11]))
            words = sent.split(' ')
            pred = [(words[i], i) for i in range(rels, rele)]
            args = [[(words[i], i) for i in range(s, e)] 
                for s, e in [(arg0s, arg0e), (arg1s, arg1e)]]
            pred_str = pred_to_openie4(pred, words)
            args_str = [arg_to_openie4(arg, words) for arg in args]
            if len(args_str) <= 0 or len(pred) <= 0:
                useless_n_sam += 1
                continue
            fout.write('{}\t\t{}\t{}\t{}\t{}\n'.format(
                0, args_str[0], pred_str, ';'.join(args_str[1:]), sent))
    print('totally {} useless samples'.format(useless_n_sam))
