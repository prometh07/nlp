import math, collections, itertools


class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    self.vocabularySize = 0 # the number of unique unigrams
    self.numberOfTokens = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        self.numberOfTokens += len(sentence.data)
        self.bigramCounts[sentence.data[0].word][0] += 1
        for first, second in itertools.izip(sentence.data, sentence.data[1:]):
            self.bigramCounts[first.word][second.word] += 1
            self.bigramCounts[second.word][0] += 1
    self.vocabularySize = len(self.bigramCounts)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for first, second in itertools.izip(sentence, sentence[1:]):
        if self.bigramCounts[first][second] > 0:
            score += math.log(self.bigramCounts[first][second])
            score -= math.log(self.bigramCounts[first][0])
        else:
            score += math.log(0.4 * (self.bigramCounts[second][0] + 1))
            score -= math.log(self.numberOfTokens + self.vocabularySize)
    return score
