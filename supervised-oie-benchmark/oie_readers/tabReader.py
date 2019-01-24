""" Usage:
    tabReader --in=INPUT_FILE

Read a tab-formatted file.
Each line consists of:
sent, prob, pred, arg1, arg2, ...

"""

from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction
from docopt import docopt
import logging

logging.basicConfig(level = logging.DEBUG)

class TabReader(OieReader):

    def __init__(self):
        self.name = 'TabReader'

    def read(self, fn):
        """
        Read a tabbed format line
        Each line consists of:
        sent, prob, pred, arg1, arg2, ...
        """
        d = {}
        ex_index = 0
        with open(fn) as fin:
            for line in fin:
                if not line.strip():
                    continue
                data = line.strip().split('\t')
                text, confidence, rel = data[:3]
                rel = rel.split('##')
                pred_pos = int(rel[1]) if len(rel) == 2 else None
                rel = rel[0]
                curExtraction = Extraction(pred=rel,
                                           pred_pos=pred_pos,
                                           head_pred_index=None,
                                           sent = text,
                                           confidence = float(confidence),
                                           question_dist = "./question_distributions/dist_wh_sbj_obj1.json",
                                           index = ex_index)
                ex_index += 1

                for arg in data[3:]:
                    arg = arg.split('##')
                    arg_pos = int(arg[1]) if len(arg) == 2 else None
                    arg = arg[0]
                    curExtraction.addArg(arg, arg_pos)

                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d


if __name__ == "__main__":
    args = docopt(__doc__)
    input_fn = args["--in"]
    tr = TabReader()
    tr.read(input_fn)
