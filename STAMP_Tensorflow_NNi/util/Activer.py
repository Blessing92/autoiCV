#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def activer(inputs, case='tanh'):
    '''
    The active enter. 
    '''
    switch = {
        'tanh': tanh,
        'relu': relu,
        'sigmoid': sigmoid,
    }
    func = switch.get(case, tanh)
    return func(inputs)


def tanh(inputs):
    '''
    The tanh active. 
    '''
    return tf.nn.tanh(inputs)

def sigmoid(inputs):
    '''
    The sigmoid active.
    '''
    return tf.nn.sigmoid(inputs)


def relu(inputs):
    '''
    The relu active. 
    '''
    return tf.nn.relu(inputs)
