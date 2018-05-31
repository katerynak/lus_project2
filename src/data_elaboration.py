import pandas as pd
import embeddings as emb


def load(filename):
    """
    loads file and stores it in pandas frame object: each line will contain
    sentence iob_tags

    :param filename:
    :return:
    """

    data = pd.read_csv(filename, sep='\s+', header=None, skip_blank_lines=False)

    token_sentence = []
    tag_sentence = []
    token_sentences = []
    tag_sentences = []

    for tok, tag in zip(data[0], data[1]):
            if pd.isnull(tok):
                tag_sentences.append(' '.join(tag_sentence))
                token_sentences.append(' '.join(token_sentence))
                tag_sentence = []
                token_sentence = []
            else:
                tag_sentence.append(tag)
                token_sentence.append(tok)

    return pd.DataFrame({'tokens': token_sentences, 'tags': tag_sentences})


def data_distr(s):
    """
    returns distribution of words in data series
    :param s: data series to analyze
    :return:
    """
    counts = pd.Series(' '.join(s).split()).value_counts()
    return counts


if __name__== "__main__":
    d = load("../data/NLSPARQL.train.data")
    print(d)
