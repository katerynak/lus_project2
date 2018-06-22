from data_elaboration import get_vocab
import gensim
import pandas as pd
import numpy as np


def filter_w2v(files, w2v):
    """

    :param files:
    :param w2v:
    :return:
    """

    vocab = set()
    for file in files:
        vocab |= get_vocab(file)
    vocab &= set(w2v.vocab)

    res = pd.DataFrame()
    tokens = []
    vectors = []

    weights = w2v.syn0
    for token in vocab:
        tokens.append(token)
        vectors.append(weights[w2v.vocab[token].index])
    res["token"] = tokens
    res["vector"] = vectors

    return res


def get_w2v(w2vfile, filename=None):
    """
    creates pandas w2v wrapper with index set to words and elements containing vectors with weights
    :param w2vfile: pickle file
    :return: pandas DataFrame
    """
    w2v_df = pd.read_pickle(w2vfile)

    # w2v_df_lowercase = w2v_df
    # w2v_df_lowercase.token = w2v_df_lowercase.token.str.lower()
    #
    # w2v_df = w2v_df.append(w2v_df_lowercase)

    # add embeddings for unknown words and padding
    unk_emb = np.zeros(len(w2v_df.vector[0]))
    pad_emb = np.zeros(len(w2v_df.vector[0]))

    if filename is not None:
        vocab = get_vocab(filename)
        for word in vocab:
            word = str(word)
            if w2v_df.token[w2v_df.token.isin([word])].empty:
                if w2v_df.token[w2v_df.token.isin([word.capitalize()])].empty:
                    w2v_df = w2v_df.append({'token': word, 'vector': pad_emb}, ignore_index=True)

    w2v_df = w2v_df.append({'token': 'UNK', 'vector': unk_emb}, ignore_index=True)
    w2v_df = w2v_df.append({'token': 'PAD', 'vector': pad_emb}, ignore_index=True)

    # create dataframe with index containing words
    w2v = pd.DataFrame(data=w2v_df.vector.tolist())
    w2v.index = w2v_df.token

    return w2v


def get_w2v_w2id(w2v):
    """
    returns word to index dictionary of embeddings
    :param w2v: word2vec dataframe
    :return:
    """
    return dict(zip(w2v.index, list(range(len(w2v.index)))))


if __name__ == "__main__":
    # writing pickle file
    datafiles = ["../data/NLSPARQL.train.data", "../data/NLSPARQL.test.data"]
    # TODO: put original file
    google_w2vfile = "../embeddings/GoogleNews-vectors-negative300.bin"
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(google_w2vfile, binary=True)
    filtered_w2v = filter_w2v(datafiles, embedding_model)
    filtered_w2v.to_pickle("../embeddings/w2v_trimmed.pickle")

    #reading pickle file
    w2vfile = "../embeddings/w2v_trimmed.pickle"
    w2v = get_w2v(w2vfile)
    # get embedding of word
    word = "much"
    emb = w2v.loc[word]
    print("embedding of word \"{}\": {}".format(word, emb))
    print("embedding values: {}".format(w2v.values))
    print("embedding words: {}".format(w2v.index))
