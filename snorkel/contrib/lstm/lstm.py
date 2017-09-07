import os
import numpy as np

import warnings

from snorkel.learning.disc_learning import TFNoiseAwareModel
from utils import *
from six.moves.cPickle import dump, load

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils

from snorkel.learning.utils import reshape_marginals, LabelBalancer


class LSTM(TFNoiseAwareModel):
    name = 'LSTM'
    representation = True
    gpu = ['gpu', 'GPU']

    """LSTM for relation extraction"""

    def _preprocess_data(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
            # Add paddings
            map(self.word_dict.get, ['~~[[1', '1]]~~', '~~[[2', '2]]~~'])
        data = []
        for candidate in candidates:
            # Mark sentence
            args = [
                (candidate[0].get_word_start(), candidate[0].get_word_end(), 1),
                (candidate[1].get_word_start(), candidate[1].get_word_end(), 2)
            ]
            s = mark_sentence(candidate_to_tokens(candidate), args)
            # Either extend word table or retrieve from it
            f = self.word_dict.get if extend else self.word_dict.lookup
            data.append(np.array(map(f, s)))
        return data

    def _check_max_sentence_length(self, ends, max_len=None):
        """Check that extraction arguments are within @self.max_len"""
        mx = max_len or self.max_sentence_length
        for i, end in enumerate(ends):
            if end >= mx:
                w = "Candidate {0} has argument past max length for model:"
                info = "[arg ends at index {0}; max len {1}]".format(end, mx)
                warnings.warn('\t'.join([w.format(i), info]))

    def load_dict(self):
        # load dict from file
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()

        # Add paddings
        map(self.word_dict.get, ['~~[[1', '1]]~~', '~~[[2', '2]]~~'])

        # Word embeddings
        f = open(self.word_emb_path, 'r')

        l = list()
        for _ in f:
            if len(_.strip().split(' ')) > self.word_emb_dim + 1:
                l.append(' ')
            else:
                word = _.strip().split(' ')[0]
                # Replace placeholder to original word defined by user.
                for key in self.replace.keys():
                    word = word.replace(key, self.replace[key])
                l.append(word)
        map(self.word_dict.get, l)
        f.close()

    def load_embeddings(self):
        self.load_dict()
        # Random initial word embeddings
        self.word_emb = np.random.uniform(-0.1, 0.1, (self.word_dict.s, self.word_emb_dim)).astype(np.float)

        # Word embeddings
        f = open(self.word_emb_path, 'r')

        for line in f:
            line = line.strip().split(' ')
            if len(line) > self.word_emb_dim + 1:
                line[0] = ' '
            for key in self.replace.keys():
                line[0] = line[0].replace(key, self.replace[key])
            self.word_emb[self.word_dict.lookup_strict(line[0])] = np.asarray(
                [float(_) for _ in line[-self.word_emb_dim:]])
        f.close()

    def train_model(self, model, optimizer, criterion, x, y):
        batch_size, max_sent = x.size()
        state_word = model.init_hidden(batch_size)
        optimizer.zero_grad()
        if self.host_device in self.gpu:
            x = x.cuda()
            y = y.cuda()
            state_word = (state_word[0].cuda(), state_word[1].cuda())
        y_pred = model(x.transpose(0, 1), state_word)
        if self.host_device in self.gpu:
            loss = criterion(y_pred.cuda(), y)
        else:
            loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data[0]

    def _init_kwargs(self, **kwargs):

        self.model_kwargs = kwargs

        # Set use pre-trained embedding or not
        self.load_emb = kwargs.get('load_emb', False)

        # Set word embedding dimension
        self.word_emb_dim = kwargs.get('word_emb_dim', 300)

        # Set word embedding path
        self.word_emb_path = kwargs.get('word_emb_path', None)

        # Set learning rate
        self.lr = kwargs.get('lr', 1e-3)

        # Set use attention or not
        self.attention = kwargs.get('attention', True)

        # Set dropout
        self.dropout = kwargs.get('dropout', 0.0)

        # Set learning epoch
        self.n_epochs = kwargs.get('n_epochs', 100)

        # Set learning batch size
        self.batch_size = kwargs.get('batch_size', 100)

        # Set rebalance setting
        self.rebalance = kwargs.get('rebalance', False)

        # Set max sentence length
        self.max_sentence_length = kwargs.get('max_sentence_length', 100)

        # Set host device
        self.host_device = kwargs.get('host_device', 'cpu')

        # Replace placeholders in embedding files
        self.replace = kwargs.get('replace', {})

        print "==============================================="
        print "Number of learning epochs:     ", self.n_epochs
        print "Learning rate:                 ", self.lr
        print "Use attention                  ", self.attention
        print "dropout                        ", self.dropout
        print "Batch size:                    ", self.batch_size
        print "Rebalance:                     ", self.rebalance
        print "Load pre-trained embedding     ", self.load_emb
        print "Host device                    ", self.host_device
        print "Word embedding size:           ", self.word_emb_dim
        print "Word embedding:                ", self.word_emb_path
        print "==============================================="

        if self.load_emb:
            assert self.word_emb_path is not None

    def train(self, X_train, Y_train, X_dev=None, Y_dev=None, rebalance=False, print_freq=5, max_sentence_length=None,
              **kwargs):

        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """

        self._init_kwargs(**kwargs)

        # Set random seed
        torch.manual_seed(self.seed)

        cardinality = Y_train.shape[1] if len(Y_train.shape) > 1 else 2
        if cardinality != self.cardinality:
            raise ValueError("Training marginals cardinality ({0}) does not"
                             "match model cardinality ({1}).".format(Y_train.shape[1],
                                                                     self.cardinality))
        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)
        # Make sure marginals are in [0,1] (v.s e.g. [-1, 1])
        if self.cardinality > 2 and not np.all(Y_train.sum(axis=1) - 1 < 1e-10):
            raise ValueError("Y_train must be row-stochastic (rows sum to 1).")
        if not np.all(Y_train >= 0):
            raise ValueError("Y_train must have values in [0,1].")

        if self.cardinality == 2:
            # This removes unlabeled examples and optionally rebalances
            train_idxs = LabelBalancer(Y_train).get_train_idxs(self.rebalance,
                                                               rand_state=self.rand_state)
        else:
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            train_idxs = np.where(diffs > 1e-6)[0]
        X_train = [X_train[j] for j in train_idxs] if self.representation \
            else X_train[train_idxs, :]
        Y_train = Y_train[train_idxs]

        print "[%s] n_train= %s" % (self.name, len(X_train))

        X_train = self._preprocess_data(X_train, extend=True)
        if X_dev is not None:
            X_dev = self._preprocess_data(X_dev, extend=False)

        if self.load_emb:
            # load embeddings from file
            self.load_embeddings()

            print "Done loading pre-trained embeddings..."

        X_train = np.array(X_train)
        Y_train = torch.from_numpy(Y_train).float()

        X = torch.from_numpy(np.arange(len(X_train)))
        data_set = data_utils.TensorDataset(X, Y_train)
        train_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=False)

        n_classes = 1 if self.cardinality == 2 else None
        self.model = AttentionRNN(n_classes=n_classes, batch_size=self.batch_size, num_tokens=self.word_dict.s,
                                  embed_size=self.word_emb_dim,
                                  lstm_hidden=100,
                                  attention=self.attention,
                                  dropout=self.dropout,
                                  bidirectional=True)
        if self.load_emb:
            self.model.lookup.weight.data.copy_(torch.from_numpy(self.word_emb))

        if self.host_device in self.gpu:
            self.model.cuda()

        n_examples = len(X_train)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss = nn.MultiLabelSoftMarginLoss(size_average=False)

        for idx in range(self.n_epochs):
            cost = 0.
            for x, y in train_loader:
                x = pad_batch(X_train[x.numpy()], self.max_sentence_length)
                y = Variable(y.float(), requires_grad=False)
                cost += self.train_model(self.model, optimizer, loss, x, y)
            if (idx + 1) % print_freq == 0:
                msg = "[%s] Epoch %s, Training error: %s" % (self.name, idx + 1, cost / n_examples)
                print msg

    def _marginals_batch(self, X):
        X_w = self._preprocess_data(X, extend=False)
        X_w = np.array(X_w)
        sigmoid = nn.Sigmoid()

        y = np.array([])

        x = torch.from_numpy(np.arange(len(X_w)))
        data_set = data_utils.TensorDataset(x, x)
        data_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=False)

        for x, _ in data_loader:
            x_w = pad_batch(X_w[x.numpy()], self.max_sentence_length)
            batch_size, max_sent = x_w.size()
            w_state_word = self.model.init_hidden(batch_size)
            if self.host_device in self.gpu:
                x_w = x_w.cuda()
                w_state_word = (w_state_word[0].cuda(), w_state_word[1].cuda())
            y_pred = self.model(x_w.transpose(0, 1), w_state_word)
            y = np.append(y, sigmoid(y_pred).data.numpy())
        return y


    def save(self, model_name=None, save_dir='checkpoints', verbose=True,
             global_step=0):
        """Save current model."""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_kwargs.pkl"), 'wb') as f:
            dump(self.model_kwargs, f)

        # Save model dicts needed to rebuild model
        with open(os.path.join(model_dir, "model_dicts.pkl"), 'wb') as f:
            dump({'word_dict': self.word_dict, 'word_emb': self.word_emb}, f)

        torch.save(self.model, os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        # Load model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_kwargs.pkl"), 'rb') as f:
            model_kwargs = load(f)
            self._init_kwargs(**model_kwargs)

        # Save model dicts needed to rebuild model
        with open(os.path.join(model_dir, "model_dicts.pkl"), 'rb') as f:
            d = load(f)
            self.word_dict = d['word_dict']
            self.word_emb = d['word_emb']

        self.model = torch.load(os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Loaded model <{1}>".format(self.name, model_name))
