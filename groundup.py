# New Monte Cristo generator I'm building from Tensorflow.

import numpy as np
import tensorflow as tf
import os
import pickle
import customlstm
from customlstm import Lstm
from tensorflow.python import debug as tf_debug
import datetime

seq_length = 100
num_units = 256
num_epochs = 20
mat_fname = 'cmonte.npy'
p_fname = 'cmonte.dat'
learn_rate = 0.001

def oh_encode(arr, num_bins=None, num_rows=None):
    if not num_bins:
        num_bins = max(arr)
    if not num_rows:
        num_rows = len(arr)
    mat = np.zeros((num_rows, num_bins), dtype=np.float32)
    for i, v in enumerate(arr):
        mat[i,v-1] = 1
    return mat

# Read and process the text
if (os.path.isfile(mat_fname) and os.path.isfile(p_fname)):
    #oh_text = np.load(mat_fname)
    with open (p_fname, 'rb') as pf:
        chars, uchars, text_len, m_text = pickle.load(pf)
else:
    raw_text = ''
    print('Reading file.')
    with open('countmonte.txt', encoding='utf8') as f:
        raw_text = f.read().lower()

    print('Processing file.')
    chars = sorted(list(set(raw_text)))
    uchars = len(chars)
    print('Number of unique characters: ' + str(uchars))
    char_to_int = {c : i for i, c in enumerate(chars)}
    int_to_char = {i : c for i, c in enumerate(chars)}

    text_len = len(raw_text)

    m_text = [char_to_int[c] for c in raw_text]
    del raw_text
    #oh_text = oh_encode(m_text, num_bins = uchars)
    #np.save(mat_fname, oh_text)
    with open(p_fname, 'wb') as pf:
        pickle.dump((chars, uchars, text_len, m_text), pf)

num_batches = (text_len // seq_length) * (seq_length)
dataIn = []
dataOut = []
for i in range(text_len - seq_length):
    seq_in = m_text[i:i+seq_length]
    #seq_out = oh_text[i+seq_length:i+seq_length+1,:]
    seq_out = m_text[i+seq_length]
    dataIn.append(seq_in)
    dataOut.append(seq_out)

n_patterns = len(dataIn)
del m_text

print('Number of patterns: {:}'.format(n_patterns))

print('Formatting data...')
# Here we have the extra dimension so that unstack results in shapes of (1,1)
procIn = np.reshape(dataIn, (n_patterns, seq_length, 1, 1))
procIn = procIn / float(uchars)
#procIn = tf.constant(procIn, dtype=tf.float32)
#procOut = oh_encode(dataOut, num_bins=uchars)
procOut = np.reshape(dataOut, (n_patterns, 1))
#procOut = tf.constant(procOut, dtype=tf.int32)
print('Data formatted.')


in_shape = list(procIn.shape[:])
in_shape[0] = None
out_shape = list(procOut.shape[:])
out_shape[0] = None
print('in_shape: {:}\nout_shape: {:}'.format(in_shape, out_shape))
inp_ph = tf.placeholder(tf.float32, in_shape, name='input_placeholder')
otp_ph = tf.placeholder(tf.int32, out_shape, name='output_placeholder')
print('Preparing input dataset')
# This section can take HUGE amount of RAM, which may result in Python / Tensorflow bailing without warning
inp_dataset = tf.data.Dataset.from_tensor_slices(inp_ph)
print('Input data shape: {:}'.format(inp_dataset.output_shapes))
print('Input data type: {:}'.format(inp_dataset.output_types))
inp_iterator = inp_dataset.make_initializable_iterator()
inp = inp_iterator.get_next()
#del procIn # Don't need this big array anymore, get some heap back
otp_dataset = tf.data.Dataset.from_tensor_slices(otp_ph)
print('Output data shape: {:}'.format(otp_dataset.output_shapes))
otp_iterator = otp_dataset.make_initializable_iterator()
otp = otp_iterator.get_next()
#del procOut # Don't need this one either, get more heap back

print('Preparing model')
us_inp = tf.unstack(inp)

lstm = Lstm(num_units, us_inp[0].get_shape())

Ws = tf.Variable(np.random.rand(num_units, uchars), dtype=tf.float32, name='Out_categorize')
bs = tf.Variable(np.zeros((1, uchars)), dtype=tf.float32, name='Categorize_bias')

st_ci, st_mi = lstm.zero_state()
o = []
for i in range(seq_length):
    o_i, st_ci, st_mi = lstm.call(us_inp[i], st_ci, st_mi)

o_i = tf.nn.dropout(o_i, 0.3)
o_i = tf.add(tf.matmul(tf.transpose(o_i), Ws, name='out_categorization'), bs, name='out_cat_add_bias')
    #o_i = tf.nn.softmax(tf.matmul(tf.transpose(o_i), Ws) + bs)
    #o.append(o_i)
o = o_i
sm = tf.nn.softmax(o, name='test_softmax')

#o = tf.concat(o, 0)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=otp, logits=o)
losses = tf.reduce_mean(loss, name='loss_mean')
optimizer = tf.train.AdamOptimizer(learn_rate)
#train_step = optimizer.compute_gradients(losses)
#apply_step = optimizer.apply_gradients(train_step, name='apply_gradients')
train_step = optimizer.minimize(losses)
#train_step = tf.contrib.slim.learning.create_train_op(losses, optimizer,summarize_gradients=True)

saver = tf.train.Saver()

mergesummary = tf.summary.merge_all()

print('Beginning the session.')
with tf.Session() as sess:
    print('Initializing variables')
    sess.run(tf.global_variables_initializer())
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #writer = tf.summary.FileWriter('log/', sess.graph)
    print('Initializing iterators.')
    sess.run(inp_iterator.initializer, feed_dict={inp_ph:procIn})
    sess.run(otp_iterator.initializer, feed_dict={otp_ph:procOut})
    print('About to train.')
    for ep in range(num_epochs):
        print('Starting epoch {:}'.format(ep))
        ptime = datetime.datetime.now()
        ttime = datetime.timedelta()
        run_loss = []
        for bat in range(num_batches):
            _train_step, _loss = sess.run([train_step, losses])
            #writer.add_summary(_summ)
            run_loss.append(_loss)
            if bat > 0 and bat % 1000 == 0:
                ctime = datetime.datetime.now()
                aloss = np.median(run_loss)
                minloss = np.min(run_loss)
                maxloss = np.max(run_loss)
                print('+Loss at batch {:} of {:}: {:} ({:} to {:})'.format(bat, batch_size, aloss, minloss, maxloss))
                dtime = ctime - ptime
                ttime += dtime
                print('|\tTime elapsed (current set / total): {:} / {:}'.format(str(dtime), str(ttime)))
                remaining = (batch_size - bat) * (dtime / 1000)
                print('|\tEstimated epoch duration (remaining / total): {:} / {:}   '.format(str(remaining), str(ttime+remaining)))
                print('|\tEstimated time of epoch completion: {:}'.format((datetime.datetime.now() + remaining).ctime()))
                run_loss = []
                ptime = datetime.datetime.now()
        print('Checkpoint created: {:}'.format(saver.save(sess, 'saves/gup.ckpt')))
    #writer.close()
