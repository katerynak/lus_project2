import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_elaboration import seq_batch


torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, tagset_size, device, pretrained_embeddings=False,
                 vocab_size=None, w2v_weights=None, bidirectional=False, num_layers=1, drop_rate=0.7, freeze=False):
        super(LSTMTagger, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        # dropout – If non-zero, introduces a Dropout layer
        # on the outputs of each LSTM layer except the last layer,
        # with dropout probability equal to dropout. Default: 0
        # bidirectional – If True, becomes a bidirectional LSTM. Default: False

        # save params
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = device
        if pretrained_embeddings:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
            self.word_embeddings.max_norm = 6
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.drop_rate = drop_rate
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.drop = nn.Dropout(self.drop_rate)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim // (1 if not bidirectional else 2),
                            dropout=self.drop_rate, batch_first=True, num_layers=self.num_layers,
                            bidirectional=bidirectional)

        if bidirectional:
            self.init_hidden = self.__init_hidden_bidirectional

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.ReLU(inplace=True)
        )

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # from pytorch documentation: (hidden state (h): (num_layers, mini_batch_size, hidden_dim),
        #                                cell state (c): (num_layers, mini_batch_size, hidden_dim)

        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device))

    def __init_hidden_bidirectional(self, batch_size):
        """
        hidden layer of bidirectional lstm: 2 hidden layers each of dimension
        """
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2, device=self.device),
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2, device=self.device))

    def forward(self, batch):
        hidden = self.init_hidden(len(batch))
        data, labels = seq_batch(batch)
        embeds = self.word_embeddings(data)
        embeds = self.drop(embeds)
        """
        Inputs: input, (h_0, c_0)
            input of shape (seq_len, batch, input_size): 
            tensor containing the features of the input sequence. 
            The input can also be a packed variable length sequence.
            h_0: tensor containing the initial hidden state for each element in the batch.
            c_0: tensor containing the initial cell state for each element in the batch.
            default values of h_0 and c_0 are zeros
        Outputs: output, (h_n, c_n)
            output of shape (seq_len, batch, num_directions * hidden_size): 
            tensor containing the output features (h_t) from the last layer of the LSTM, for each t
            h_n of shape (num_layers * num_directions, batch, hidden_size): 
                tensor containing the hidden state for t = seq_len
            c_n (num_layers * num_directions, batch, hidden_size): 
                tensor containing the cell state for t = seq_len

        """
        lstm_out, hidden = self.lstm(embeds, hidden)
        # send output to fc layer(s)
        tag_space = self.hidden2tag(lstm_out.unsqueeze(1).contiguous())
        tag_scores = F.log_softmax(tag_space, dim=3)
        return tag_scores.view(-1, self.tagset_size), labels.view(-1)


class LSTMTagger_seq(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger_seq, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # from pytorch documentation: (hidden state (h): (num_layers, mini_batch_size, hidden_dim),
        #                                cell state (c): (num_layers, mini_batch_size, hidden_dim)

        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
