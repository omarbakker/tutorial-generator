import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import os
import time
import math
from utils import *

SEQ_LEN = 30
BATCH_SIZE = 200
INTERNAL_SIZE = 512
N_LAYERS = 3
α = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout


if __name__ == '__main__':

    # load our data
    dataset = ""
    with open('data/dataset.txt') as fullFile:
        dataset = fullFile.read()
    if len(dataset) == 0: print("Error, failed to load data")

    # get the data, use 10% for validation!
    encodings, mapping, reverseMapping = encodeCharacters(dataset)
    import pdb; pdb.set_trace()
    
    valSplit = (len(encodings)*9)//10
    Xval = encodings[valSplit:]
    Xtrn = encodings[:valSplit]
    ALPHABET_SIZE = len(mapping) + 1

    epoch_size = len(Xtrn) // (BATCH_SIZE * SEQ_LEN)
    print("Training char count: ", len(Xtrn))
    print("Validation char count: ", len(Xval))
    print("Epoch size ", epoch_size)

    # hyper-parameters
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    # inputs
    X = tf.placeholder(tf.uint8, [None, None], name='X')
    Xo = tf.one_hot(X, ALPHABET_SIZE, 1.0, 0.0)
    Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')
    Yo_ = tf.one_hot(Y_, ALPHABET_SIZE, 1.0, 0.0)
    Hin = tf.placeholder(tf.float32, [None,INTERNAL_SIZE*N_LAYERS], name='Hin')

    # cells/network
    cells = [rnn.GRUCell(INTERNAL_SIZE) for _ in range(N_LAYERS)]
    cells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
    cells = rnn.MultiRNNCell(cells, state_is_tuple=False)
    Yr, H = tf.nn.dynamic_rnn(cells, Xo, dtype=tf.float32, initial_state=Hin)
    H = tf.identity(H, name='H')

    # Softmax Layer
    Yflat = tf.reshape(Yr, [-1, INTERNAL_SIZE])
    Ylogits = layers.linear(Yflat, ALPHABET_SIZE)
    Yflat_ = tf.reshape(Yo_, [-1, ALPHABET_SIZE])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)
    loss = tf.reshape(loss, [batchsize, -1])
    Yo = tf.nn.softmax(Ylogits, name='Yo')
    Y = tf.argmax(Yo, 1)
    Y = tf.reshape(Y, [batchsize, -1], name="Y")
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)),
                              tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    # Tensorboard stuff
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

    # Init for saving models.
    # Only the last checkpoint is kept.
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1000)

    DISPLAY_FREQ = 50
    _50_BATCHES = DISPLAY_FREQ * BATCH_SIZE * SEQ_LEN
    progress = Progress(DISPLAY_FREQ, size=111+2,
                            msg="Training on next "+str(DISPLAY_FREQ)+" batches")

    # init
    istate = np.zeros([BATCH_SIZE, INTERNAL_SIZE*N_LAYERS])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0

    # training loop
    for x, y_, epoch in\
            rnn_minibatch_sequencer(Xtrn, BATCH_SIZE, SEQ_LEN, nb_epochs=10):

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, Hin: istate, lr: α,
                     pkeep: dropout_pkeep, batchsize: BATCH_SIZE}
        _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

        # log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCH_SIZE}  # no dropout for validation
            y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
            print_learning_learned_comparison(x, y, mapping, l, bl, acc, epoch_size, step, epoch)
            summary_writer.add_summary(smm, step)

        # Adapted from: https://github.com/martin-gorner/tensorflow-rnn-shakespeare
        # run a validation step every 50 batches
        # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
        # so we cut it up and batch the pieces (slightly inaccurate)
        # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
        if step % _50_BATCHES == 0 and len(Xval) > 0:
            VALI_SEQLEN = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
            bsize = len(Xval) // VALI_SEQLEN
            print_validation_header(len(Xtrn))
            vali_x, vali_y, _ = next(rnn_minibatch_sequencer(Xval, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, INTERNAL_SIZE*N_LAYERS])
            feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,  # no dropout for validation
                         batchsize: bsize}
            ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            print_validation_stats(ls, acc)
            # save validation data for Tensorboard
            validation_writer.add_summary(smm, step)

        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            print_text_generation_header()
            ry = np.array([[reverseMapping.get(ord('<'))]])
            rh = np.zeros([1, INTERNAL_SIZE * N_LAYERS])
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2, ALPHABET_SIZE=ALPHABET_SIZE)
                print(chr(mapping.get(rc)), end="")
                ry = np.array([[rc]])
            print_text_generation_footer()

        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
            print("Saved file: " + saved_file)

        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)

        # loop state around
        istate = ostate
        step += BATCH_SIZE * SEQ_LEN




# end
