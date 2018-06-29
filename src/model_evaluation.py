from train import train_evaluate
import itertools
import numpy as np

if __name__== "__main__":

    #modelarc = ["lstm", "bilstm", "gru", "bigru", "rnn", "birnn", "jordan", "bijordan"]
    modelarc = ["bilstm"]
    train = ["../data/NLSPARQL.train"]
    dev = "../data/NLSPARQL.dev"
    test = ["../data/NLSPARQL.test.data"]
    freeze = [False]
    drop = 0.7
    epochs = 50
    hidden_size = 200
    outfile = "../models_out/model"

    emb = ["pretrained", "train"]
    learning_rates = [0.005, 0.001]
    # batch_sizes = [1, 5, 20, 40, 100, 300]
    batch_sizes = [ 8, 10, 15, 20, 25]
    drops = [0.6, 0.7, 0.8]
    epochs = [25, 30, 35, 40, 45]
    # hidden_sizes = [150, 200, 300]
    hidden_sizes = [200, 300, 400]
    embedding_sizes = [300]
    decays = [0.0]
    word_unk_probs = [0, 0.0005, 0.001, 0.01]
    tag_unk_probs = [0, 0.0005, 0.001, 0.01]

    params = [modelarc, train, test, batch_sizes, learning_rates, decays, emb,
                           embedding_sizes, freeze, drops, epochs, hidden_sizes, word_unk_probs, tag_unk_probs]

    params = list(itertools.product(*params))
    params = np.array(params)
    iterations = 10
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for modelarc, train, test, batch_size, learning_rate, decay,\
        emb, embedding_size, freeze, drop, epochs, hidden_size, \
        word_unk_prob, tag_unk_prob in params:
        if freeze:
            if emb == "train":
                continue

        model = train_evaluate(modelarc, train, dev, test, int(batch_size),
                               float(learning_rate), float(decay), emb,
                               int(embedding_size), False,
                               float(drop), int(epochs), int(hidden_size),
                               float(word_unk_prob), float(tag_unk_prob))



       # model = train_evaluate(*param)
    #
    # model = train_evaluate(modelarc, train, test, batch_size, lr, emb,
    #                        embedding_size, freeze, drop, epochs, hidden_size, )