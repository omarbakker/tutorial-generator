import numpy as np

def encodeCharacters(text, COUNT_THRESHOLD=20):
    '''
    Encode the text file's characters and return an array of encodings as well
    as a mapping of the characters to encodings to be used for decoding.
    Chars with < COUNT_THRESHOLD occurences will be mapped to 0 to simplify the
    encodings, all other chars will be encoded as their index position in a
    sorted set of the characters
    '''
    encodings = np.array([ord(c) for c in text])
    unique = set(encodings)
    charCounts = dict(zip(list(unique),\
                [len(encodings[encodings == u]) for u in list(unique)]))
    copy = np.array(encodings)
    for c in encodings:
        if c == 0: continue
        count = charCounts[c]
        if count < COUNT_THRESHOLD:
            encodings[encodings == c] = 0
            unique.remove(c)
    mapping = dict(zip(range(1,len(unique)+1), sorted(list(unique))))
    reverseMapping = dict(zip(sorted(list(unique)), range(1,len(unique)+1)))
    for i in range(len(encodings)):
        if encodings[i] in reverseMapping:
            encodings[i] = reverseMapping.get(encodings[i])
    return encodings, mapping, reverseMapping


def decodeCharacters(encodings, mapping):
    '''
    Returns the original set of characters encoded with encode characters,
    characters that we're encoded as 0 or unknown are returned as a ?
    '''
    chars = []
    for char in encodings:
        if char in mapping:
            char = chr(mapping.get(char))
        else:
            char = "?"
        chars.append(char)
    return "".join(chars)


def sample_from_probabilities(probabilities, topn, ALPHABET_SIZE):
    """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.
    :param probabilities: a list of size ALPHASIZE with individual probabilities
    :param topn: the number of highest probabilities to consider. Defaults to all of them.
    :return: a random integer
    """
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(ALPHABET_SIZE, 1, p=p)[0]


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Adapted from: https://github.com/martin-gorner/tensorflow-rnn-shakespeare
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch

def print_learning_learned_comparison(X, Y, mapping, losses, batch_loss, batch_accuracy, epoch_size, index, epoch):
    """
    Adapted from: https://github.com/martin-gorner/tensorflow-rnn-shakespeare
    Display utility for printing learning statistics
    """
    print()
    # epoch_size in number of batches
    batch_size = X.shape[0]  # batch_size in number of sequences
    sequence_len = X.shape[1]  # sequence_len in number of characters
    start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        decx = decodeCharacters(X[k], mapping)
        decy = decodeCharacters(Y[k], mapping)
        epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
        loss_string = "loss: {:.5f}".format(losses[k])
        print_string = epoch_string + " │ {} │ {} │ {}"
        print(print_string.format(decx, decy, loss_string))
        index += sequence_len
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510
    format_string = "└{:─^" + str(len(epoch_string)) + "}"
    format_string += "┴{:─^" + str(len(decx) + 2) + "}"
    format_string += "┴{:─^" + str(len(decy) + 2) + "}"
    format_string += "┴{:─^" + str(len(loss_string)) + "}┘"
    footer = format_string.format('INDEX', 'TRAINING SEQUENCE', 'PREDICTED SEQUENCE', 'LOSS')
    print(footer)
    # print statistics
    batch_index = start_index_in_epoch // (batch_size * sequence_len)
    batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
    stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(batch_string, batch_loss, batch_accuracy)
    print()
    print("TRAINING STATS: {}".format(stats))


class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress

def print_validation_header(validation_start):
    print('Validating')


def print_validation_stats(loss, accuracy):
    print("VALIDATION STATS:                                  loss: {:.5f},       accuracy: {:.5f}".format(loss,
                                                                                                           accuracy))

def print_text_generation_header():
    print()
    print("┌{:─^111}┐".format('Generating random text from learned state'))


def print_text_generation_footer():
    print()
    print("└{:─^111}┘".format('End of generation'))


def frequency_limiter(n, multiple=1, modulo=0):
    def limit(i):
        return i % (multiple * n) == modulo*multiple
    return limit
