from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction
from _collections import defaultdict
from matcher import Matcher

class GoldReader(OieReader):
    
    # Path relative to repo root folder
    default_filename = './oie_corpus/all.oie' 
    
    def __init__(self):
        self.name = 'Gold'
    
    def read(self, fn):
        d = defaultdict(lambda: [])
        with open(fn) as fin:
            for line_ind, line in enumerate(fin):
                data = line.strip().split('\t')
                text, rel = data[:2]
                args = data[2:]
                confidence = 1
                
                curExtraction = Extraction(pred = rel,
                                           head_pred_index = None,
                                           sent = text,
                                           confidence = float(confidence),
                                           index = line_ind)
                for arg in args:
                    curExtraction.addArg(arg)
                    
                d[text].append(curExtraction)
        self.oie = d

    def extract_head(self, fn, out_fn):
        def get_range(sent, frag):
            sent_pad = ' ' + sent + ' '
            frag_pad = ' ' + frag + ' '
            start_ind = sent_pad.find(frag_pad)
            if start_ind < 0:
                print('can\'t find \n{}\nin \n{}'.format(frag, sent))
                raise Exception('wrong sentence')
            frag_word = frag.split(' ')
            start_ind = len([w for w in sent[:start_ind].split(' ') if w != ''])
            end_ind = start_ind + len(frag_word)
            return start_ind, end_ind
        stanford_parse_result = {}
        with open(fn) as fin, open(out_fn, 'w') as fout:
            for line_ind, line in enumerate(fin):
                data = line.strip().split('\t')
                sent, pred = data[:2]
                args = data[2:]
                confidence = 1

                sent_split = sent.split(' ')
                heads = []
                for frag in [pred] + args:
                    st, ed = get_range(sent, frag)
                    if sent not in stanford_parse_result:
                        stanford_parse_result[sent] = Matcher.stanford_parse(sent_split)
                    dl = stanford_parse_result[sent]
                    head = Matcher.get_syntactic_head(sent_split, st, ed, depth_list=dl)
                    heads.append(head)
                fout.write('{}\t{}\t{}\t{}\n'.format(sent, pred, '\t'.join(args), '\t'.join(heads)))
        

if __name__ == '__main__' :
    g = GoldReader()
    g.read('../oie_corpus/all.oie', includeNominal = False)
    d = g.oie
    e = d.items()[0]
    print e[1][0].bow()
    print (g.count())
