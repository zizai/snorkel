import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

softmax = nn.Softmax()
sigmoid = nn.Sigmoid()


class SymbolTable(object):
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2, unknown_symbol=1):
        self.s       = starting_symbol
        self.unknown = unknown_symbol
        self.d       = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, self.unknown)

    def lookup_strict(self, w):
        return self.d.get(w)

    def len(self):
        return self.s

    def reverse(self):
        return {v: k for k, v in self.d.iteritems()}


def scrub(s):
    return ''.join(c for c in s if ord(c) < 128)


def candidate_to_tokens(candidate, token_type='words'):
    tokens = candidate.get_parent().__dict__[token_type]
    return [scrub(w).lower() for w in tokens]


def mark(l, h, idx):
    """Produce markers based on argument positions

    :param l: sentence position of first word in argument
    :param h: sentence position of last word in argument
    :param idx: argument index (1 or 2)
    """
    return [(l, "{}{}".format('~~[[', idx)), (h + 1, "{}{}".format(idx, ']]~~'))]


def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence

    :param s: list of tokens in sentence
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments

    Example: Then Barack married Michelle.
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x

def pad_batch(mini_batch, max_len):
    mini_batch_size = len(mini_batch)
    max_sent_len = min(int(np.max([len(x) for x in mini_batch])), max_len)
    # print mini_batch_size, max_sent_len
    main_matrix = np.zeros((mini_batch_size, max_sent_len), dtype= np.int)
    for idx1, i in enumerate(mini_batch):
        for idx2, j in enumerate(i):
            if idx2 >= max_sent_len: break
            main_matrix[idx1, idx2] = j
    main_matrix = Variable(torch.from_numpy(main_matrix))
    return main_matrix


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if (nonlinearity == 'tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if (s is None):
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if (nonlinearity == 'tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if (attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0)


class AttentionRNN(nn.Module):
    def __init__(self, n_classes, batch_size, num_tokens, embed_size, lstm_hidden, bidirectional=True):

        super(AttentionRNN, self).__init__()

        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.n_classes = n_classes

        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            self.word_gru = nn.LSTM(embed_size, lstm_hidden, bidirectional=True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2 * lstm_hidden, 2 * lstm_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2 * lstm_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2 * lstm_hidden, 1))
            self.linear = nn.Linear(2 * lstm_hidden, n_classes)
        else:
            self.word_gru = nn.LSTM(embed_size, lstm_hidden, bidirectional=False)
            self.weight_W_word = nn.Parameter(torch.Tensor(lstm_hidden, lstm_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(lstm_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(lstm_hidden, 1))
            self.linear = nn.Linear(lstm_hidden, n_classes)


        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

    def forward(self, embed, state_word):
        # embeddings
        embedded = self.lookup(embed)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1, 0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1, 0))
        final_map = self.linear(word_attn_vectors)
        return final_map

    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.lstm_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.lstm_hidden))
