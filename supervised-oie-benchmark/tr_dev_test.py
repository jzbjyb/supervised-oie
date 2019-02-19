import argparse
import pandas as pd
from operator import itemgetter
from oie_readers.extraction import Extraction
from conll_to_openie4 import arg_to_openie4, pred_to_openie4

def get_sent_set(filepath):
    sents = []
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l == '':
                continue
            sents.append(l)
    return set(sents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split into train dev test')
    parser.add_argument('-inp', type=str, help='input file')
    parser.add_argument('-out', type=str, help='output file')
    opt = parser.parse_args()

    train = 'raw_sentences/train.txt'
    dev = 'raw_sentences/dev.txt'
    test = 'raw_sentences/test.txt'
    trs = get_sent_set(train)
    devs = get_sent_set(dev)
    tes = get_sent_set(test)

    with open(opt.inp, 'r') as fin, \
        open(opt.out + 'train.txt', 'w') as tr_out, \
        open(opt.out + 'dev.txt', 'w') as dev_out, \
        open(opt.out + 'test.txt', 'w') as te_out:
        for l in fin:
            l = l.strip()
            if l == '':
                continue
            s = l.split('\t')[-1]
            if s in trs:
                tr_out.write('{}\n'.format(l))
            elif s in devs:
                dev_out.write('{}\n'.format(l))
            elif s in tes:
                te_out.write('{}\n'.format(l))
            else:
                print('out of set')
                print(l)
