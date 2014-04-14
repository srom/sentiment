# -*- coding: utf-8 -*-

import re
from nltk.corpus import movie_reviews
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import numpy as np


FEATURES_NUMBER = 1000
NGRAMS_NUMBER = 2

#==== Gradient descent constants
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
        # TODO split by sentences
        # TODO stemmize
        # remove irrelevant stop words
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
        Compute the nb features of a given document.
         - most common words: 1 if the document contains this word, else 0
         - most common bigrams: 1 if the document contains this bigram, else 0

         Args:
            document: The document as a string.

        Returns:
            A list of binary features.
        """
        features = []

        # most common ngrams for n = 1 to 2
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


    def train(self, speed):
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
        print '==== (Stochastic Gradient Descent)'
        self.w = gradient_descent(x_train, y_train, x_val, y_val, speed)
        print '==== Done'
        return self.w


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
        return 1 if sigmoid(np.dot(np.transpose(self.w), x)) >= 0 else 0


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
        x: documents matrix (numpy.array)
        w: weight vector (numpy.array)
        y: output vector (numpy.array)
        h: function of x and w

    Returns:
        The cost value (float).
    """
    n = x.shape[0]
    val = 0
    for i in xrange(n):
        val += (y[i] * np.log(h(x[i], w))
            + (1.0 - y[i]) * np.log(1.0 - h(x[i], w)))
    return -1.0 * (val / n)

def gradient_descent(train_set, y_train, test_set, y_test, speed):
    """
    Stochastic gradient descent.

    Args:
        train_set: the train set (numpy.array)
        y_train: the training output vector (numpy.array)
        test_set: the test set (numpy.array)
        y_test: the testing output vector (numpy.array)
        speed: The speed of the descent.

    Returns:
        The weight vector which minimize the logistic cost function (numpy.array)
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
            x = train_set[i]
            for j in xrange(m):
                w[j] = w[j] - speed * (h(x, w) - y_train[i]) * x[j]

        # test convergence
        mse = cost(test_set, w, y_test, h)
        if iteration > 1:
            err_diff = abs(100 - (last_err / mse) * 100)
        last_err = mse
        print 'MSE: %f' % mse
        print 'DIFF: {0} %'.format(err_diff)
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

    sentiment_scores = np.array([1 if category == 'neg' else 0 
        for category in movie_reviews.categories() 
        for review_id in movie_reviews.fileids(category)])

    #== select random indices
    n = documents.shape[0]
    indices = np.random.permutation(n)
    threshold = np.floor(n*0.8)
    train_idx, test_idx = indices[:threshold], indices[threshold:]

    #== select training and validation sets according to these indicies
    x_train, x_test = documents[:, train_idx], documents[:, test_idx]
    y_train, y_test = sentiment_scores[:, train_idx], sentiment_scores[:, test_idx]

    #== train the model
    print '===== Training the model...'
    sentiment = SentimentMachine(x_train.tolist(), y_train.tolist())
    w = sentiment.train(0.001)
    print '===== Model trained.'

    #== test the model
    print '===== Testing the model...'
    # compute the MSE
    h = lambda a,b: sigmoid(np.dot(a,b))
    x = sentiment.compute_features_matrix(x_test.tolist())
    mse = cost(x, w, y_test, h)
    # compute the number of valid classifications
    n_test = y_test.shape[0]
    valid = 0
    for i in xrange(n_test):
        valid += 1 if sentiment.test(x_test[i]) == y_test[i] else 0
    percent = 100.0 * valid / n_test
    # print results
    print ('== Number of well-classified documents: {0} / {1} ({2}%)'
        .format(valid, n_test, percent))
    print '== MSE on the test set: %f' % mse


if __name__ == '__main__':
    main()
