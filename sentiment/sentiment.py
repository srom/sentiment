# -*- coding: utf-8 -*-

import re
from nltk.corpus import movie_reviews
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import numpy as np

#==== gradient descent constants
MAX_ITERATIONS = 100
THRESHOLD_CONVERGENCE = 0.5 # in percentage
THRESHOLD_DIVERGENCE = 5 # in percentage
#=====

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
        # dictionnary of sets of ngrams
        # { 1: set('word1', 'word2', ...), 2: set('bi gram1', 'bi gram2', ...) }
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
        # lowercase and tokenize
        tokens = word_tokenize(document.lower())
        # remove punctuation
        tokens = [word for word in tokens if re.search('[a-z0-9]+', word)]
        if n == 1:
            # return the list of words
            return tokens
        else:
            # return the list of ngrams
            return ngrams(tokens, n)

    def get_most_common_ngrams(self, n, nb_ngrams):
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
        print '== Compute %d-grams' % n
        all_ngrams = []
        for document in self.training_set:
            all_ngrams.extend(self.compute_ngrams(document, n))

        # get the frequency
        freq = FreqDist(ngram for ngram in all_ngrams)
        # store and return the nb_ngrams most common ngrams
        self._most_common_ngrams[n] = freq.keys()[:nb_ngrams]
        return self._most_common_ngrams[n]


    def document_features(self, document):
        """
        Compute the 6000 features of a given document.
         - 2000 most common words: 1 if the document contains this word, else 0
         - 2000 most common bigrams: 1 if the document contains this bigram, else 0
         - 2000 most common trigrams: 1 if the document contains this trigram, else 0

         Args:
            document: The document as a string.

        Returns:
            A list of 6000 elements.
        """
        features = []

        common_ngrams = []
        ngrams = set(self.compute_ngrams(document, 2))
        for ngram in self.get_most_common_ngrams(2, 2000):
            common_ngrams.append(1 if ngram in ngrams else 0)
        features.extend(common_ngrams)

        # most common ngrams for n = 1 to 3
        # for n in range(3):
        #     common_ngrams = []
        #     ngrams = set(self.compute_ngrams(document, n+1))
        #     for ngram in self.get_most_common_ngrams(n+1, 2000):
        #         common_ngrams.append(1 if ngram in ngrams else 0)
        #     features.extend(common_ngrams)

        return features


    def load_train_matrix(self):
        """
        Load the Nx6000 train matrix X where N is equals to the number of 
        documents in the training set.

        Returns:
            A Nx6000 matrix (numpy.matrix) 
        """
        m = []
        count = 0
        total = len(self.training_set)
        for document in self.training_set:
            count += 1
            print '=== Document %d / %d' % (count, total)
            m.append(self.document_features(document))
        return np.array(m)


    def train(self, speed):
        """
        Train the model via logistic regression (stochastic gradient descent).

        Args:
            speed: Speed of the gradient descent.
        """
        # load training matrix
        print '==== Load training matrix...'
        x = self.load_train_matrix()
        y = np.transpose(np.array(self.score_set))
        print '==== Done'

        # select random indices
        [n,m] = x.shape
        indices = np.random.permutation(n)
        threshold = np.floor(n*0.8)
        train_idx, val_idx = indices[:threshold], indices[threshold:]

        # select training and validation sets according to these indicies
        x_train, x_val = x[train_idx,:], x[val_idx,:]
        y_train, y_val = y[train_idx, :], y[val_idx, :]

        # train like a boss
        print '==== Start training...'
        self.w = gradient_descent(x_train, y_train, x_val, y_val, speed)
        print '==== Done'


    def test(self, test_string):
        """
        Test the logistic model on the given string.

        Args:
            test_string: the test string.

        Returns:
            The predicted output value. 
        """
        if self.w is None:
            raise ValueError('Looks like you forgot to .train() ' 
                + 'the model before .test()-ing it!')

        # get features vector
        x = self.document_features(test_string)

        # compute h(transpose(w) * x) and return the result according
        # to the boundary h(transpose(w) * x) = 0
        return 1 if sigmoid(np.dot(np.transpose(w), x)) >= 0 else 0


def sigmoid(z):
    """
    The sigmoid / logistic function.

    Args:
        z: any real number.

    Returns:
        A value between O and 1.
    """
    return 1.0 / (1.0 + np.exp(-1.0 * z))

def cost(x, w, y, h):
    """
    Cost function of the logistic regression.

    Args:
        x: documents matrix (numpy.matrix)
        w: weight vector (numpy.matrix)
        y: output vector (numpy.matrix)
        h: function of x and w.

    Returns:
        The cost value (float).
    """
    [n,m] = x.shape
    val = 0
    for i in xrange(n):
        val += (y[i] * np.log(h(x[i], w))
            + (1.0 - y[i]) * np.log(1.0 - h(x[i], w)))
    return -1.0 * (val / n)

def gradient_descent(train_set, y_train, test_set, y_test, speed):
    """
    Stochastic gradient descent.

    Args:
        train_set: the train set (numpy.matrix)
        y_train: the training output vactor (numpy.matrix)
        test_set: the test set (numpy.matrix)
        y_test: the testing output vactor (numpy.matrix)
        speed: The speed of the descent.

    Returns:
        The weight vector which minimize the logistic cost function (numpy.matrix)
    """
    # get the dimensions of the train set
    [n,m] = train_set.shape
    # init the weight vector
    w = np.zeros(m)
    # init variables
    iteration = 0
    err_diff = THRESHOLD_CONVERGENCE + 1
    last_err = 0
    # define h as the sigmoid of transpose(w[i]) * x[i]
    h = lambda a,b: sigmoid(np.dot(a,b))
    # stochastic gradient descent
    while (
        iteration < MAX_ITERATIONS
        and err_diff > THRESHOLD_CONVERGENCE
        # and err_diff < THRESHOLD_DIVERGENCE
    ):
        iteration += 1
        print 'iteration %d...' % iteration

        # compute w
        for i in xrange(n):
            # print '%d / %d' % (i, n)
            x = train_set[i]
            for j in xrange(m):
                w[j] = w[j] - speed * (h(x, w) - y_train[i]) * x[j]

        # test convergence
        mse = cost(test_set, w, y_test, h)
        if iteration > 1:
            err_diff = abs(100 - (last_err / mse) * 100)
        last_err = mse
        print 'MSE: %f' % mse
        print 'DIFF: {} %'.format(err_diff)
        print

    return w


def main():
    """
    Sample train using the movie review corpus (Pang, Lee).
    """
    documents = np.array([movie_reviews.raw(review_id) 
        for category in movie_reviews.categories() 
        for review_id in movie_reviews.fileids(category)])

    sentiment_scores = np.array([1 if category == 'neg' else 0 
        for category in movie_reviews.categories() 
        for review_id in movie_reviews.fileids(category)])

    # select random indices
    n = documents.shape[0]
    indices = np.random.permutation(n)
    threshold = np.floor(n*0.8)
    train_idx, test_idx = indices[:threshold], indices[threshold:]

    # select training and validation sets according to these indicies
    x_train, x_test = documents[:, train_idx], documents[:, test_idx]
    y_train, y_test = sentiment_scores[:, train_idx], sentiment_scores[:, test_idx]

    # GO!
    sentiment = SentimentMachine(x_train.tolist(), y_train.tolist())
    sentiment.train(0.001)


if __name__ == '__main__':
    main()
