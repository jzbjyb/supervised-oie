""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

Convert to tabbed format
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
import re

# Local imports
from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

#=-----

class OpenieFourReader(OieReader):

    def __init__(self):
        self.name = 'OpenIE-4'

    @staticmethod
    def parse_openie4str(sent, openie4str):
        substr = openie4str[openie4str.index('(') + 1:openie4str.index(',List(')]
        char_start_ind = openie4str[openie4str.index(',List(') + 7:]
        char_start_ind = int(re.split(',|}', char_start_ind)[0])
        return (substr, OpenieFourReader.find_word_range(sent, substr, char_start_ind))

    @staticmethod
    def find_word_range(sent, substr, char_start_ind):
        len_in_word = len(substr.strip().split(' '))
        st = len(sent[:char_start_ind].split(' ')) - 1
        return [st + i for i in range(len_in_word)]

    def read(self, fn):
        d = {}
        d_list = []
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                confidence = data[0]
                if not all(data[2:5]):
                    logging.debug("Skipped line: {}".format(line))
                    continue
                text = data[5]
                #arg1, rel, arg2 = [s[s.index('(') + 1:s.index(',List(')] for s in data[2:5]]
                arg1, rel, arg2 = [OpenieFourReader.parse_openie4str(text, s) for s in data[2:5]]
                head_pred_index = rel[1][0] # TODO: head_pred_index is not necessarily the first predicate index
                curExtraction = Extraction(pred=rel, head_pred_index=head_pred_index, sent=text, confidence=float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                if text not in d:
                    d[text] = []
                    d_list.append([])
                d[text].append(curExtraction)
                d_list[-1].append(curExtraction)
        self.oie = d
        self.oie_list = d_list



if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)


    oie = OpenieFourReader()
    oie.read(inp_fn)
    oie.output_tabbed(out_fn)

    logging.info("DONE")
