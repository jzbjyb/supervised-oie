import argparse
from oie_readers.extraction import Extraction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combine conll')
    parser.add_argument('-inp', type=str, help='input conll files separated by ":"')
    parser.add_argument('-gold', type=str, help='gold conll file', default=None)
    parser.add_argument('-out', type=str, help='output conll file')
    args = parser.parse_args()

    f1, f2 = args.inp.split(':')

    Extraction.combine_conll(f1, f2, args.out, gold_fn=args.gold)