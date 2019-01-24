import numpy as np

class TaggingPredict():
    '''
    An autoregressive prediction interface for tagging model
    '''
    def __init__(self, prob):
        self.prob = prob
        self.sent_len = prob.shape[0]
        self.num_class = prob.shape[1]

    def __call__(self, live_samples):
        ret = []
        for s in live_samples:
            if len(s) >= self.prob.shape[0]:
                # reach the end of the sentence
                return None
            ret.append(self.prob[len(s)])
        return np.array(ret)

def beamsearch(predict, k=1):
    '''
    return k samples (beams) and their NLL scores, each sample is a sequence of labels
    '''
    live_samples = [[]] # start with empty sequence
    live_scores = [0]

    while True:
        # for every possible live sample calc prob for every possible label
        probs = predict(live_samples)

        if probs is None:
            break

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        cand_flat = cand_scores.flatten()

        # find the best (lowest) scores we have from all possible samples and new words
        ranks_flat = cand_flat.argsort()[:(k)]
        live_scores = cand_flat[ranks_flat]
        cur_prob = probs.flatten()

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r // voc_size] + [(r % voc_size, cur_prob[r])] for r in ranks_flat]

    return live_samples, live_scores
