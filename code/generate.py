import tensorflow as tf
import numpy as np
import utils

INTERNAL_SIZE = 512
N_LAYERS = 3

tutor = "checkpoints/rnn_train_1510734663-78000000"
mapdir = 'data/mapping.txt'
rmapdir = 'data/reverseMapping.txt'

author = tutor

ncnt = 0
with tf.Session() as sess, open(mapdir) as mapping, open(rmapdir) as reverseMapping:

    mapping = mapping.readlines()
    reverseMapping = reverseMapping.readlines()
    mapping = {l.split(':')[0]:l.split(':')[1].replace('\n','') for l in mapping}
    reverseMapping = {l.split(':')[0]:l.split(':')[1].replace('\n','') for l in reverseMapping}
    ALPHABET_SIZE = len(mapping) + 1

    new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1510734663-78000000.meta')
    new_saver.restore(sess, author)

    x = reverseMapping.get(str(ord('<')))
    print('<', end="")
    x = np.array([[x]])

    # initial values
    y = x
    h = np.zeros([1, INTERNAL_SIZE * N_LAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
    for i in range(1000000000):

        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

        c = utils.sample_from_probabilities(yo, topn=10, ALPHABET_SIZE=ALPHABET_SIZE)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(int(mapping.get(str(c))))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 100:
            print("")
            ncnt = 0
