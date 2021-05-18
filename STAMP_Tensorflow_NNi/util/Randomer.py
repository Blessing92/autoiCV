#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

class Randomer(object):
    stddev = None

    @staticmethod
    def random_normal(wshape):
        return tf.random_normal(wshape, stddev=Randomer.stddev)

    @staticmethod
    def set_stddev(sd):
        Randomer.stddev = sd