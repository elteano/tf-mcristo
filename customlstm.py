# Because the library one is being mean.

import tensorflow as tf
import numpy as np

class Lstm(object):
    """ LSTM implementation based on
    https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
    """
    def __init__(self, num_units, input_shape):
        self.__num_units = num_units
        self.__input_elems = input_shape.num_elements()
        self.__scope = tf.name_scope('lstm')
        # Input weight initialization
        with self.__scope:
            self.__Wf = self.__gen_input_weight(name='Wf')
            self.__Wi = self.__gen_input_weight(name='Wi')
            self.__Wo = self.__gen_input_weight(name='Wo')
            self.__Wc = self.__gen_input_weight(name='Wc')
            # Recurrent weight initialization
            self.__Uf = self.__gen_rec_weight(name='Uf')
            self.__Ui = self.__gen_rec_weight(name='Ui')
            self.__Uo = self.__gen_rec_weight(name='Uo')
            self.__Uc = self.__gen_rec_weight(name='Uc')
            # Bias initialization
            self.__Bf = self.__gen_bias(name='Bf')
            self.__Bi = self.__gen_bias(name='Bi')
            self.__Bo = self.__gen_bias(name='Bo')
            self.__Bc = self.__gen_bias(name='Bc')

    def __gen_input_weight(self, name=None):
        return tf.Variable(np.random.rand(self.__num_units, self.__input_elems) / self.__num_units, dtype=tf.float32, name=name)

    def __gen_rec_weight(self, name=None):
        return tf.Variable(np.random.rand(self.__num_units, self.__num_units) / self.__num_units, dtype=tf.float32, name=name)

    def __gen_bias(self, name=None):
        return tf.Variable(np.zeros((self.__num_units, 1)), dtype=tf.float32, name=name)

    def call(self, input, prev_state, prev_out):
        with self.__scope:
            with tf.name_scope('call'):
                x = tf.reshape(input, (self.__input_elems, 1), name='Input_reshape')
                f = tf.sigmoid(tf.add_n([tf.matmul(self.__Wf, x, name='Wf_matmul'), tf.matmul(self.__Uf, prev_out, name='Uf_matmul'), self.__Bf], name='forget_sum'), name='forget_sigmoid')
                i = tf.sigmoid(tf.add_n([tf.matmul(self.__Wi, x, name='Wi_matmul'), tf.matmul(self.__Ui, prev_out, name='Ui_matmul'), self.__Bi], name='i_sum'), name='i_sigmoid')
                o = tf.sigmoid(tf.add_n([tf.matmul(self.__Wo, x, name='Wo_matmul'), tf.matmul(self.__Uo, prev_out, name='Uo_matmul'), self.__Bo], name='o_sum'), name='o_sigmoid')
                c_input = tf.tanh(tf.add_n([tf.matmul(self.__Wc, x, name='Wc_matmul'), tf.matmul(self.__Uc, prev_out, name='Uc_matmul'), self.__Bc], name='c_input_sum'), name='c_input_tanh')
                c = tf.add_n([tf.multiply(f, prev_state, name='prev_state_multiply'), tf.multiply(i, c_input)], name='c_final_sum')
                h = tf.multiply(o, tf.tanh(c, name='c_tanh'), name='hidden_output')
                return o, c, h

    @property
    def weights(self):
        return (self.__Wo)

    def zero_state(self):
        return (tf.constant(np.zeros((self.__num_units, 1), dtype=np.float32), dtype=tf.float32),
            tf.constant(np.zeros((self.__num_units, 1), dtype=np.float32), dtype=tf.float32))
