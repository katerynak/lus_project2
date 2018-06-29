import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.utils import shuffle
import os

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


def write_pos_iob(iob_file, pos_file, out_file):
    """
    writes file containing word	pos_tag	iob_tag
    """
    data_pos = pd.read_csv(pos_file, sep='\s+', header=None, skip_blank_lines=False)
    data_iob = pd.read_csv(iob_file, sep='\s+', header=None, skip_blank_lines=False)

    data_pos.columns = ['token', 'pos_tag', 'lemma']
    data_iob.columns = ['token', 'iob_tag']

    out_data = pd.DataFrame(data=[data_pos.token, data_pos.pos_tag, data_iob.iob_tag]).transpose()
    out_data.to_csv(out_file, index=None, header=None, sep=' ', mode='w')


def write_pref_pos_iob(iob_file, pos_file, out_file):
    """
    writes file containing word	pos_tag	iob_tag
    """
    data_pos = pd.read_csv(pos_file, sep='\s+', header=None, skip_blank_lines=False)
    data_iob = pd.read_csv(iob_file, sep='\s+', header=None, skip_blank_lines=False)

    data_pos.columns = ['token', 'pos_tag', 'lemma']
    data_iob.columns = ['token', 'iob_tag']
    data_iob['prefix'] = data_iob.token.apply(lambda x: x if (x!=x) else str(x)[:3])

    out_data = pd.DataFrame(data=[data_iob.prefix, data_pos.pos_tag, data_iob.iob_tag]).transpose()
    out_data.to_csv(out_file, index=None, header=None, sep=' ', mode='w')


def write_suff_pos_iob(iob_file, pos_file, out_file):
    """
    writes file containing word	pos_tag	iob_tag
    """
    data_pos = pd.read_csv(pos_file, sep='\s+', header=None, skip_blank_lines=False)
    data_iob = pd.read_csv(iob_file, sep='\s+', header=None, skip_blank_lines=False)

    data_pos.columns = ['token', 'pos_tag', 'lemma']
    data_iob.columns = ['token', 'iob_tag']
    data_iob['suffix'] = data_iob.token.apply(lambda x: x if (x!=x) else str(x)[-3:])

    out_data = pd.DataFrame(data=[data_iob.suffix, data_pos.pos_tag, data_iob.iob_tag]).transpose()
    out_data.to_csv(out_file, index=None, header=None, sep=' ', mode='w')


def write_prefix_iob(iob_file, out_file):
    """
    writes file containing word	prefix	iob_tag
    """
    data_iob = pd.read_csv(iob_file, sep='\s+', header=None, skip_blank_lines=False)

    data_iob.columns = ['token', 'iob_tag']
    data_iob['prefix'] = data_iob.token.apply(lambda x: x if (x!=x) else str(x)[:3])

    out_data = pd.DataFrame(data=[data_iob.token,data_iob['prefix'], data_iob.iob_tag]).transpose()
    out_data.to_csv(out_file, index=None, header=None, sep=' ', mode='w')


def write_suffix_iob(iob_file, out_file):
    """
    writes file containing word	prefix	iob_tag
    """
    data_iob = pd.read_csv(iob_file, sep='\s+', header=None, skip_blank_lines=False)

    data_iob.columns = ['token', 'iob_tag']
    data_iob['suffix'] = data_iob.token.apply(lambda x:  x if (x!=x) else str(x)[-3:])

    out_data = pd.DataFrame(data=[data_iob.token, data_iob['suffix'], data_iob.iob_tag]).transpose()
    out_data.to_csv(out_file, index=None, header=None, sep=' ', mode='w')


def load_words(filename):
    """
    loads file and stores it in pandas frame object: each line will contain
    word iob_tag, each sentence is separated from another by NaN value
    :param filename:
    :return:
    """

    data = pd.read_csv(filename, sep='\s+', header=None, skip_blank_lines=False)
    data.columns = ['x', 'y_true']
    return data


def get_vocab(filename):
    """
    extracts vocabulary given file
    :param filename:
    :return:
    """
    vocab = pd.Series([])
    data = load_words(filename)
    for c in data:
        vocab = vocab.append(pd.Series(data[c].unique()))
    vocab = vocab.unique()
    return set(vocab)


def data_distr(s):
    """
    returns distribution of words in data series
    :param s: data series to analyze
    :return:
    """
    counts = pd.Series(' '.join(s).split()).value_counts()
    return counts


def transform(sample, sentence_transform, tags_transform):
    """
    :param sample: input sample to be elaborated, dictionary with keys 'sentence', 'tags'
    :param sentence_transform: function to be applied to sentence, embedding for example
    :param tags_transform: function to be applied to tags
    :return: transformed sample
    """
    sample['sentence'] = sentence_transform(sample['sentence'])
    sample['tags'] = tags_transform(sample['tags'])
    return sample


def seq_batch(batch):
    """
    given a batch returns tensor of concatenated data
    :param batch:
    :return:
    """
    list_of_data_tensors = [sample['sentence'].unsqueeze(0) for sample in batch]
    data = torch.cat(list_of_data_tensors, dim=0).cuda()
    list_of_labels_tensors = [sample['tags'].unsqueeze(0) for sample in batch]
    labels = torch.cat(list_of_labels_tensors, dim=0).cuda()
    # char_data = None
    # if "chars" in batch[0]:
    #     list_of_char_data_tensors = [sample["chars"] for sample in batch]
    #     char_data = torch.cat(list_of_char_data_tensors, dim=0).cuda()

    return data, labels#, char_data


def test_dev_split(train_rate=0.8, trainFile="../data/NLSPARQL.train.data"):

    """
    function splits test and train file in different files to be used for cross validation,
    split is based on number of phrases and not number of lines
    """

    data = pd.read_csv(trainFile, sep="\s+", header=None, skip_blank_lines=False)
    data.columns = ['tokens', 'tags']
    delimiters = []
    delim = [0]
    tmp = pd.isnull(data).any(1).nonzero()[0].tolist()
    for d in tmp:
        delim.append(d)
        delimiters.append(delim)
        delim = [d]

    # shuffle phrases
    delimiters = shuffle(delimiters)

    # count test phrases
    train_size = int(len(delimiters)*train_rate)

    train_delimiters = delimiters[0:train_size]
    dev_delimiters = delimiters[train_size:]

    train_data = pd.DataFrame()
    dev_data = pd.DataFrame()

    for id in train_delimiters:
        train_data = train_data.append(data.loc[id[0]:id[1] - 1, :], ignore_index=True)

    for id in dev_delimiters:
        dev_data = dev_data.append(data.loc[id[0]:id[1] - 1, :], ignore_index=True)

    train_file = "../data/NLSPARQL.train"
    dev_file = "../data/NLSPARQL.dev"

    train_data.iloc[1:].to_csv(train_file, index=None, header=None, sep='\t', mode='w')
    dev_data.iloc[1:].to_csv(dev_file, index=None, header=None, sep='\t', mode='w')

    with open(train_file, mode='a') as f:
        f.write('\n')

    with open(dev_file, mode='a') as f:
        f.write('\n')


def count_tokens(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    data.columns = ['tokens', 'tags']
    tok_counts = pd.DataFrame(data['tokens'].value_counts().reset_index())
    tok_counts.columns = ['tokens', 'tokens_counts']


def count_tags(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    data.columns = ['tokens', 'tags']
    pos_counts = pd.DataFrame(data['tags'].value_counts().reset_index())
    pos_counts.columns = ['tags', 'tags_counts']


def print_dists( inputFile, filelabel=""):
    """
    plots distribution of current dataFrame data
    """
    import matplotlib.pyplot as plt

    data = pd.read_csv(inputFile, sep='\t', header=None)
    data.columns = ['tokens', 'tags']

    dir = "../plots/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    fig = plt.figure()
    dist = data['tags'].value_counts(normalize=False)
    dist.index = [x[2:] if x != 'O' else x for x in dist.index]
    dist = dist.groupby(level=0).sum().sort_values(ascending=False)
    plt.xlabel("Concepts")
    plt.ylabel("Frequency")
    dist[1:].plot(kind='bar')
    plt.tight_layout()
    plt.show()
    fig.savefig(dir + filelabel + "_tags_dist" + ".pdf", format='pdf')

    fig = plt.figure()
    dist = data['tokens'].value_counts(normalize=False)
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    dist[:30].plot(kind='bar')
    plt.tight_layout()
    # plt.show()
    fig.savefig(dir + filelabel + "_tokens_dist" + ".pdf", format='pdf')


def unk_count(self, testFile, trainFile):
    train = pd.read_csv(trainFile, sep='\t', header=None)
    train.columns = ['tokens', 'tags']

    test = pd.read_csv(testFile, sep='\t', header=None)
    test.columns = ['tokens', 'tags']

    train_tokens = train['tokens'].unique().tolist()
    test_tokens = test['tokens'].unique().tolist()

    unk_unique = list(set(test_tokens) - set(train_tokens))

    unk = [x for x in test['tokens'].tolist() if x not in train_tokens]

    train_tags = train['tags'].unique().tolist()
    test_tags = test['tags'].unique().tolist()

    unk_unique_tag = list(set(test_tags) - set(train_tags))

    unk_tag = [x for x in test['tags'].tolist() if x not in train_tags]




class Wrapped_dataset(Dataset):
    """
    wraps dataset with pytorch Dataset class
    """

    def __init__(self, data, transform=None):
        """

        :param data: pandas dataframe containing sentences and tags
        :param transform: function that transforms sentences and tags values
        """

        self.data = data
        self.transform = transform
        self.samples = []
        maxlen = 0
        for idx in range(len(self.data)):
            sentence = self.data.tokens[idx]
            tags = self.data.tags[idx]
            sample = {'sentence': sentence, 'tags': tags}
            if self.transform:
                sample = self.transform(sample)
            if len(sample['tags']) > maxlen:
                maxlen = len(sample['tags'])
            self.samples.append(sample)
        # padding
        for idx in range(len(self.data)):
            self.samples[idx]['sentence'] = F.pad(self.samples[idx]['sentence'], (0, maxlen-len(self.samples[idx]['sentence'])), 'constant', 0)
            self.samples[idx]['tags'] = F.pad(self.samples[idx]['tags'],
                                                  (0, maxlen - len(self.samples[idx]['tags'])), 'constant', -1)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.samples[idx]


if __name__== "__main__":
    d = load("../data/NLSPARQL.train.data")
    print(d)

    iob_file = "../data/NLSPARQL.train.data"
    pos_file = "../data/NLSPARQL.train.feats.txt"
    out_file = "../data/NLSPARQL.train.iob.pref.data"
    write_prefix_iob(iob_file, "../data/NLSPARQL.train.iob.pref.data")
    write_suffix_iob(iob_file, "../data/NLSPARQL.train.iob.suff.data")

    write_pref_pos_iob(iob_file, pos_file, "../data/NLSPARQL.train.pref.pos.data")
    write_suff_pos_iob(iob_file, pos_file, "../data/NLSPARQL.train.suff.pos.data")

    iob_file = "../data/NLSPARQL.test.data"
    pos_file = "../data/NLSPARQL.test.feats.txt"
    out_file = "../data/NLSPARQL.test.iob.pref.data"
    write_prefix_iob(iob_file, "../data/NLSPARQL.test.iob.pref.data")
    write_suffix_iob(iob_file, "../data/NLSPARQL.test.iob.suff.data")

    write_pref_pos_iob(iob_file, pos_file, "../data/NLSPARQL.test.pref.pos.data")
    write_suff_pos_iob(iob_file, pos_file, "../data/NLSPARQL.test.suff.pos.data")

    testFile = "../data/NLSPARQL.test.data"
    trainFile = "../data/NLSPARQL.train"
