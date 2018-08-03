# Because the library one is being mean.

import tensorflow as tf
import numpy as np

class Lstm(object):
    """ LSTM implementation based on
    https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
    """
    def __init__(self, num_units, input_shape, batch_size=1):
        self.__num_units = num_units
        self.__input_elems = input_shape[0]
        self.__scope = tf.name_scope('lstm')
        self.__batch_size = batch_size
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
                #x = tf.reshape(input, (self.__input_elems, self.__batch_size), name='Input_reshape')
                x = input
                # Forget operations
                wfm = tf.matmul(self.__Wf, x, name='Wf_matmul')
                ufm = tf.matmul(self.__Uf, prev_out, name='Uf_matmul')
                fm = tf.add(wfm, ufm, name='forget_mat_add')
                fb = tf.add(fm, self.__Bf, name='forget_bias_add')
                f = tf.sigmoid(fb, name='forget_sigmoid')
                # I operations
                wim = tf.matmul(self.__Wi, x, name='Wi_matmul')
                uim = tf.matmul(self.__Ui, prev_out, name='Ui_matmul')
                im = tf.add(wim, uim, name='i_mat_add')
                ib = tf.add(im, self.__Bi, 'i_bias_add')
                i = tf.sigmoid(ib, name='i_sigmoid')
                # O operations
                wom = tf.matmul(self.__Wo, x, name='Wo_matmul')
                uom = tf.matmul(self.__Uo, prev_out, name='Uo_matmul')
                om = tf.add(wom, uom, name='o_mat_add')
                ob = tf.add(om, self.__Bo, name='o_bias_add')
                o = tf.sigmoid(ob, name='o_sigmoid')
                # C operations
                wcm = tf.matmul(self.__Wc, x, name='Wc_matmul')
                ucm = tf.matmul(self.__Uc, prev_out, name='Uc_matmul')
                cm = tf.add(wcm, ucm, name='c_mat_add')
                cb = tf.add(wcm, self.__Bc, name='c_bias_add')
                c_tanh = tf.tanh(cb, name='c_tanh')
                c = tf.add(tf.multiply(f, prev_state, name='prev_state_multiply'), tf.multiply(i, c_tanh, name='c_tanh_mul'), name='c_final_sum')
                # h
                h = tf.multiply(o, tf.tanh(c, name='c_tanh'), name='hidden_output')
                return o, c, h

    @property
    def weights(self):
        return (self.__Wo)

    def zero_state(self):
        return (tf.constant(np.zeros((self.__num_units, self.__batch_size), dtype=np.float32), dtype=tf.float32),
            tf.constant(np.zeros((self.__num_units, self.__batch_size), dtype=np.float32), dtype=tf.float32))
