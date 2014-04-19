# -*- coding: utf-8 -*-

import re
from nltk.corpus import movie_reviews, stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
import numpy as np


#==== General parameters
FEATURES_NUMBER = 2000
NGRAMS_NUMBER = 2
REGULARISATION = 10.0

#==== Gradient descent constants
SPEED = 0.001
MAX_ITERATIONS = 20
THRESHOLD_CONVERGENCE = 1 # in percentage

#==== Text processing constants
BLACKLIST_STOPWORDS = ['over','only','very','not','no']
ENGLISH_STOPWORDS = set(stopwords.words('english')) - set(BLACKLIST_STOPWORDS)
NEG_CONTRACTIONS = [
    (r'aren\'t', 'are not'),
    (r'can\'t', 'can not'),
    (r'couldn\'t', 'could not'),
    (r'daren\'t', 'dare not'),
    (r'didn\'t', 'did not'),
    (r'doesn\'t', 'does not'),
    (r'don\'t', 'do not'),
    (r'isn\'t', 'is not'),
    (r'hasn\'t', 'has not'),
    (r'haven\'t', 'have not'),
    (r'hadn\'t', 'had not'),
    (r'mayn\'t', 'may not'),
    (r'mightn\'t', 'might not'),
    (r'mustn\'t', 'must not'),
    (r'needn\'t', 'need not'),
    (r'oughtn\'t', 'ought not'),
    (r'shan\'t', 'shall not'),
    (r'shouldn\'t', 'should not'),
    (r'wasn\'t', 'was not'),
    (r'weren\'t', 'were not'),
    (r'won\'t', 'will not'),
    (r'wouldn\'t', 'would not'),
    (r'ain\'t', 'am not') # not only but stopword anyway
]
OTHER_CONTRACTIONS = {
    "'m": 'am',
    "'ll": 'will',
    "'s": 'has', # or 'is' but both are stopwords
    "'d": 'had'  # or 'would' but both are stopwords
}

class SentimentMachine(object):
    """
    This class train a logistic regression model to analyse the sentiment
    of a document. Sentiment is either negative (0) or positive (1).
    """

    def __init__(self, training_set, score_set):
        """
        Init the SentimentMachine with the training set.

        Args:
            training_set: A list of documents (list of strings)
            score_set: A list of sentiment scores (list of numbers)
            
            len(training_set) and len(score_set) must be equal.
        """
        self.training_set = training_set
        self.score_set = score_set
        self.stemmer = PorterStemmer()
        # dictionnary of sets of ngrams
        self._most_common_ngrams = {}
        # weight vector
        self.w = None

    def compute_ngrams(self, document, n):
        """
        Compute ngrams of the document.

        Args:
            document: The document as a string.
            n: The number of grams. Must be a positive interger.

        Returns:
            A list of ngrams.
        """
        # lowercase
        doc = document.lower()
        # TODO split by sentences for more accuracy
        # transform negative contractions (e.g don't --> do not)
        for t in NEG_CONTRACTIONS:
            doc = re.sub(t[0], t[1], doc)
        # tokenize
        tokens = word_tokenize(doc)
        # transform other contractions (e.g 'll --> will)
        tokens = [OTHER_CONTRACTIONS[token] if OTHER_CONTRACTIONS.get(token) 
                    else token for token in tokens]
        # remove punctuation
        r = r'[a-z]+'
        tokens = [word for word in tokens if re.search(r, word)]
        # remove irrelevant stop words
        tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
        # stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        if n == 1:
            # return the list of words
            return tokens
        else:
            # return the list of ngrams
            return ngrams(tokens, n)

    def get_most_common_ngrams(self, n, nb_ngrams=None):
        """
        Compute and return the set of the most common ngrams in the documents.
        This set is cached inside the object.

        Args:
            n: The number of grams. Must be a positive interger.
            nb_ngrams: The number of ngrams to return, i.e quantifying the 'most'.

        Returns:
            A list of the most common ngrams.
        """
        try:
            # return cached value
            return self._most_common_ngrams[n]
        except KeyError:
            pass

        # compute all ngrams
        all_ngrams = []
        for document in self.training_set:
            all_ngrams.extend(self.compute_ngrams(document, n))

        # get the frequency or return all ngrams
        freq = FreqDist(ngram for ngram in all_ngrams)
        # store and return the nb_ngrams most common ngrams
        if nb_ngrams:
            self._most_common_ngrams[n] = freq.keys()[:nb_ngrams]
        else:
            self._most_common_ngrams[n] = freq.keys()
        return self._most_common_ngrams[n]

    def document_features(self, document):
        """
        Compute the nb features of a given document.
         - most common words: 1 if the document contains this word, else 0
         - most common bigrams: 1 if the document contains this bigram, else 0

         Args:
            document: The document as a string.

        Returns:
            A list of binary features.
        """
        features = []

        # most common ngrams for n = 1 to NGRAMS_NUMBER
        nb_ngrams = NGRAMS_NUMBER
        nb_features = FEATURES_NUMBER / nb_ngrams
        for n in range(nb_ngrams):
            common_ngrams = []
            # get ngrams in the document
            ngrams = set(self.compute_ngrams(document, n+1))
            for ngram in self.get_most_common_ngrams(n+1, nb_features):
                # if ngram is a common one then feature = 1 else 0
                common_ngrams.append(1 if ngram in ngrams else 0)
            # add new feature
            features.extend(common_ngrams)

        return features


    def compute_features_matrix(self, train_set=None):
        """
        Load the NxM matrix X where N is equals to the number of documents 
        in the set and M is equal to the number of features.

        Args:
            train_set: A list of documents (list of strings).
                If None, self.training_set is used.

        Returns:
            A NxM matrix (numpy.array) 
        """
        m = []
        for document in train_set or self.training_set:
            m.append(self.document_features(document))
        return np.array(m)


    def train(self, speed=0.001, stochastic=False):
        """
        Train the model via logistic regression (stochastic gradient descent).

        Args:
            speed: Speed of the gradient descent.
        """
        # load training matrix
        print '==== Compute training set features...'
        x = self.compute_features_matrix()
        y = np.transpose(np.array(self.score_set))
        print '==== Done'

        # shuffle
        [n,m] = x.shape
        print 'Number of features: %d' % m
        indices = np.random.permutation(n)
        x, y = x[indices,:], y[indices, :]

        # inital value for w
        w_zero = np.zeros(m)

        # train like a boss
        print '==== Start training...'
        method = 'Stochastic' if stochastic else 'Batch'
        print '==== (%s Gradient Descent)' % method
        self.w = gradient_descent(x, y, w_zero, speed=speed, stochastic=stochastic)
        print '==== Done'
        return self.w


    def classify(self, test_string):
        """
        Test the logistic model on the given string.

        Args:
            test_string: the test string.

        Returns:
            The predicted output value. 
        """
        if self.w is None:
            raise ValueError('Looks like you forgot to .train() ' 
                + 'the model before .classify()-ing it!')

        # get features vector
        x = np.array(self.document_features(test_string))

        # compute h(transpose(w) * x) and return the result according
        # to the boundary h(transpose(w) * x) = 0.5
        return 1 if sigmoid(np.dot(np.transpose(self.w), x)) >= 0.5 else 0


def sigmoid(z):
    """
    The sigmoid / logistic function.

    Args:
        z: any real number.

    Returns:
        A value between O and 1.
    """
    return 1.0 / (1.0 + np.exp(-1.0 * z))

def cost(w, x, y, h):
    """
    Cost function of the logistic regression.

    Args:
        w: weight vector (numpy.array)
        x: documents matrix (numpy.array)
        y: output vector (numpy.array)
        h: function of x and w

    Returns:
        The cost value (float).
    """
    [n, m] = x.shape
    val = 0
    # cost
    for i in xrange(n):
        val += (y[i] * np.log(h(x[i], w))
            + (1.0 - y[i]) * np.log(1.0 - h(x[i], w)))
    # regularisation
    reg = REGULARISATION * np.dot(np.transpose(w), w) / (2.0 * n)
    return -1.0 * (val / n) + reg


def batch_descent(w, x, y, h, speed):
    """
    Compute w (Batch gradient descent).

    Args:
        w: weight vector (numpy.array)
        x: documents matrix (numpy.array)
        y: output vector (numpy.array)
        h: function of x and w

    Returns:
        The gradient vector (list of float values).
    """
    [n, m] = x.shape
    for i in xrange(m):
        reg = REGULARISATION * w[i] / n
        for j in xrange(n):
            w[i] = w[i] - speed * ((h(x[j], w) - y[j]) * x[j,i] - reg)


def stochastic_descent(w, x, y, h, speed):
    """
    Compute w (Stochastic gradient descent).

    Args:
        w: weight vector (numpy.array)
        x: documents matrix (numpy.array)
        y: output vector (numpy.array)
        h: function of x and w

    Returns:
        The gradient vector (list of float values).
    """
    [n, m] = x.shape
    for i in xrange(n):
        for j in xrange(m):
            reg = REGULARISATION * w[j] / n
            w[j] = w[j] - speed * ((h(x[i], w) - y[i]) * x[i,j] - reg)

def gradient_descent(x, y, w_zero, speed=SPEED, stochastic=False, 
                    threshold=THRESHOLD_CONVERGENCE, max_iter=MAX_ITERATIONS):
    """
    Gradient descent (either batch or stochastic) find a local minimum of a 
    function f by iteratively substract a proportion of the gradient of f.

    Args:
        x: The train set (numpy.array).
        y: The training output vector (numpy.array).
        w_zero: initial value of the parameter (numpy.array).
        speed: The speed of the descent (float).
        stochastic: Batch or Stochastic gradient descent (Boolean).
        threshold: Convergence threshold for the difference between two 
            consecutive cost function values (float, in percent).
        max_iter: Maximum number of iterations (integer).

    Returns:
        The weight vector which minimize the logistic cost function (numpy.array)
    """
    # get the dimensions of the train set
    [n,m] = x.shape
    # init the weight vector
    w = w_zero
    # init variables
    iteration = 0
    diff = threshold + 1
    last_cost_val = 0
    # define h as the sigmoid of transpose(w[i]) * x[i]
    h = lambda a,b: sigmoid(np.dot(a,b))
    # gradient descent
    while (
        iteration < max_iter
        and diff > threshold
    ):
        iteration += 1
        print 'iteration %d...' % iteration

        # compute w
        if stochastic:
            # stochastic gradient descent
            stochastic_descent(w, x, y, h, speed)
        else:
            # batch gradient descent
            batch_descent(w, x, y, h, speed)

        # check convergence
        cost_val = cost(w, x, y, h)
        if iteration > 1:
            diff = abs(100 - (last_cost_val / cost_val) * 100)
        last_cost_val = cost_val
        valid = 0
        for i in xrange(n):
            v = 1 if sigmoid(np.dot(w, x[i])) >= 0.5 else 0
            valid += 1 if v == y[i] else 0
        percent = 100.0 * valid / n

        print ('Well-classified documents: {0} / {1} ({2}%)'
            .format(valid, n, percent))
        print 'Cost value: %.4f' % cost_val
        print 'DIFF: %.4f %%' % diff
        print

    return w


def main():
    """
    Sample training using the movie reviews corpus (Pang, Lee).
    """

    #== load inputs
    documents = np.array([movie_reviews.raw(review_id) 
        for category in movie_reviews.categories() 
        for review_id in movie_reviews.fileids(category)])

    sentiment_scores = np.array([0 if category == 'neg' else 1 
        for category in movie_reviews.categories() 
        for review_id in movie_reviews.fileids(category)])

    #== select random indices
    n = documents.shape[0]
    indices = np.random.permutation(n)
    threshold = np.floor(n*0.8) # 80% training set / 20% test set
    train_idx, test_idx = indices[:threshold], indices[threshold:]

    #== select training and validation sets according to these indicies
    x_train, x_test = documents[:, train_idx], documents[:, test_idx]
    y_train, y_test = sentiment_scores[:, train_idx], sentiment_scores[:, test_idx]

    #== train the model
    print '===== Training the model...'
    sentiment = SentimentMachine(x_train.tolist(), y_train.tolist())
    w = sentiment.train(speed=0.001, stochastic=False)
    print '===== Model trained.'

    #== test efficiency of the model
    print '===== Testing the model...'
    # compute the MSE
    h = lambda a,b: sigmoid(np.dot(a,b))
    x = sentiment.compute_features_matrix(x_test.tolist())
    mse = cost(w, x, y_test, h)
    # compute the number of valid classifications
    n_test = y_test.shape[0]
    valid = 0
    for i in xrange(n_test):
        valid += 1 if sentiment.classify(x_test[i]) == y_test[i] else 0
    percent = 100.0 * valid / n_test
    # print results
    print ('== Number of well-classified documents: {0} / {1} ({2}%)'
        .format(valid, n_test, percent))
    print '== Cost value on the test set: %.4f' % mse


if __name__ == '__main__':
    main()
