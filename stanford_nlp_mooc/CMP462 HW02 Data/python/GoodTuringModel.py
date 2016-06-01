import math
import itertools
import collections
import matplotlib.pyplot as plt

class GoodTuringModel:
  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.frequenciesCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        self.total += len(sentence.data)
        for datum in sentence.data:
            self.unigramCounts[datum.word] += 1
    for frequency in self.unigramCounts.values():
        self.frequenciesCounts[frequency] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for word in sentence:
        if self.unigramCounts[word] == 0:
            score += math.log(self.frequenciesCounts[1]/float(self.total**2)) # hack 1
        else:
            c = self.unigramCounts[word]
            classFrequency = self.frequenciesCounts[c+1] or self.frequenciesCounts[c] # hack 2
            score += math.log((c + 1) * classFrequency)
            score -= math.log(self.frequenciesCounts[c])
        score -= math.log(self.total)
    return score
