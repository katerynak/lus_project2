import numpy as np
import re
import torch, torch.tensor
from nltk import ngrams
from torchtext import data


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
    # idxs = list(map(lambda x: word2id.get(x, word2id['UNK']), doc.split()))
    idxs = []
    for word in doc.split():
        idx = word2id.get(word, word2id['UNK'])
        if idx == word2id['UNK']:
            idx = word2id.get(word.capitalize(), word2id['UNK'])
            if idx == word2id['UNK']:
                print("mapped as unknown:" + word)
        idxs.append(idx)
    return torch.tensor(idxs, dtype=torch.long)


def ids2doc(ids, word2id):
    """
    creates list of words given ids
    :param ids:
    :param word2id:
    :return:
    """
    doc = list(map(lambda x: list(word2id.keys())[list(word2id.values()).index(x)], ids))
    return doc


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

    return torch.tensor(ret)


def normalize(s):
    """
    eliminates all characters which are not word characters (e.g. question marks, dots)
    :param s: data series to elaborate
    :return:
    """
    text = str(s)
    pattern = re.compile('[\W]+', re.UNICODE) # \W Matches any character which is not a word character.
    return pattern.sub(r' ', text.lower()).strip()


def doc2ngrams(s,n):
    return list(ngrams(s.split(), n))


def giveMeNameLater():
    TEXT = data.Field(lower=True, batch_first=True, fix_length=20)
    LABEL = data.Field(sequential=False)


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
