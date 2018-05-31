import numpy as np
import re


def vocabulary(s):
    """
    returns vocabulary from given pandas series containing text
    :param s: pandas series from which you want to extract vocabulary
    :return:
    """
    vocab = set()
    for doc in s:
        vocab |= set(doc.split())

    return vocab


def create_w2id(vocab):
    """
    vocabulary indexing
    :param vocab:
    :return:
    """
    w2id = {'PAD': 0, 'UNK': 1}
    for i, word in enumerate(vocab, 2):
        w2id[word] = i
    return w2id


def doc2ids(doc, word2id):
    """
    converts document to list of unique ids
    :param doc: document
    :param word2id: indexed vocabulary
    :return:
    """
    return list(map(lambda x: word2id.get(x, word2id['UNK']), doc.split()))


def doc2bow(doc, word2id):
    """
    converts document to bag of words embedding
    :param doc: sentence, text
    :param word2id: indexed vocabulary
    :return:
    """
    ret = np.zeros(len(word2id))
    doc2id = doc2ids(doc,word2id)

    for w in set(doc2id):
        ret[w] = doc2id.count(w)

    return ret


def normalize(s):
    """
    eliminates all characters which are not word characters (e.g. question marks, dots)
    :param s: data series to elaborate
    :return:
    """
    text = str(s)
    pattern = re.compile('[\W]+', re.UNICODE) # \W Matches any character which is not a word character.
    return pattern.sub(r' ', text.lower()).strip()


if __name__== "__main__":
    import data_elaboration as de

    data = de.load("../data/NLSPARQL.train.data")
    data_n = data.tokens.apply(normalize)
    chanded_idx = [data_n!=data.tokens]

    for i, j in zip(data.tokens.values[chanded_idx], data_n.values[chanded_idx]):
        print("{} --> {}".format(i, j))

    vocab = vocabulary(data.tokens)
    word2id = create_w2id(vocab)
    print(word2id)

