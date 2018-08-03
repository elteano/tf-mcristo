# Monte Cristo in Tensorflow

This is a little recurrent neural net I'm putting together as a learning foray
into Tensorflow. The LSTM units I'm using are implemented (probably incorrectly)
by hand.

This is the successor to my Keras implementation. I felt that Keras did not
provide enough flexibility for generating from recurrent networks, and so was
driven into the lower levels to get the flexibility I seek.

Currently, I have wrangled it together so that we have a single-layer LSMT
network which trains on the *Monte Cristo* data. Generation is coming soon.
