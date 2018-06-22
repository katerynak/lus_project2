import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch


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
            if len(sample['tags'])>maxlen:
                maxlen=len(sample['tags'])
            self.samples.append(sample)
        #padding
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
