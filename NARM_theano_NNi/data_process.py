from __future__ import print_function
import numpy
import theano
numpy.random.seed(42)


def prepare_data(seqs, labels):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.ones((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
    x_mask *= (1 - (x == 0))
    return x, x_mask, labels