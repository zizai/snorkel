import os
import numpy as np

import warnings

from snorkel.learning.disc_learning import TFNoiseAwareModel
from utils import candidate_to_tokens, SymbolTable
from six.moves.cPickle import dump, load

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils

from snorkel.learning.utils import reshape_marginals, LabelBalancer


def build_model(input_dim, output_dim):
    # We don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    return model


def train_model(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output


class PCA(TFNoiseAwareModel):

    name = 'PCA'
    representation = True

    """SVD for relation extraction"""

    def _preprocess_data_combination(self, candidates, extend=False):
        """Convert candidate sentences to lookup sequences

        :param candidates: candidates to process
        :param extend: extend symbol table for tokens (train), or lookup (test)?
        """
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()
        data = []
        for candidate in candidates:
            words = candidate_to_tokens(candidate)

            sent, m1, m2, word_seq = [], [], [], []

            # Word level embeddings
            sent = np.array(words)

            k = self.window_size

            m1_start = candidate[0].get_word_start()
            m1_end = candidate[0].get_word_end() + 1

            m1_start = max(m1_start - k, 0)
            m1_end = min(m1_end + k, len(words))

            m2_start = candidate[1].get_word_start()
            m2_end = candidate[1].get_word_end() + 1

            m2_start = max(m2_start - k, 0)
            m2_end = min(m2_end + k, len(words))

            m1 = np.array(words[m1_start: m1_end])
            m2 = np.array(words[m2_start: m2_end])
            st = min(candidate[0].get_word_end(), candidate[1].get_word_end())
            ed = max(candidate[0].get_word_start(), candidate[1].get_word_start())
            word_seq = np.array(words[st + 1: ed])

            order = 0 if m1_start < m2_start else 1
            data.append((sent, m1, m2, word_seq, order))

        return data

    def _check_max_sentence_length(self, ends, max_len=None):
        """Check that extraction arguments are within @self.max_len"""
        mx = max_len or self.max_len
        for i, end in enumerate(ends):
            if end >= mx:
                w = "Candidate {0} has argument past max length for model:"
                info = "[arg ends at index {0}; max len {1}]".format(end, mx)
                warnings.warn('\t'.join([w.format(i), info]))

    def _build_model(self, **model_kwargs):
        pass

    def load_dict(self):
        # load dict from glove
        if not hasattr(self, 'word_dict'):
            self.word_dict = SymbolTable()

        if self.char and not hasattr(self, 'char_dict'):
            self.char_dict = SymbolTable()

        # Word embeddins
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

        if self.char:
            # Char embeddings
            f = open(self.char_emb_path, 'r')

            l = list()
            for _ in f:
                if len(_.strip().split(' ')) > self.char_emb_dim + 1:
                    l.append(' ')
                else:
                    word = _.strip().split(' ')[0]
                    # Replace placeholder to original word defined by user.
                    for key in self.replace.keys():
                        word = word.replace(key, self.replace[key])
                    l.append(word)
            map(self.char_dict.get, l)
            f.close()

    def load_embeddings(self):
        self.load_dict()
        # Random initial word embeddings
        self.word_emb = np.random.uniform(-0.01, 0.01, (self.word_dict.s, self.word_emb_dim)).astype(np.float)

        if self.char:
            # Random initial char embeddings
            self.char_emb = np.random.uniform(-0.01, 0.01, (self.char_dict.s, self.char_emb_dim)).astype(np.float)

        # Word embeddings
        f = open(self.word_emb_path, 'r')

        for line in f:
            line = line.strip().split(' ')
            if len(line) > self.word_emb_dim + 1:
                line[0] = ' '
            for key in self.replace.keys():
                line[0] = line[0].replace(key, self.replace[key])
            self.word_emb[self.word_dict.lookup_strict(line[0])] = np.asarray([float(_) for _ in line[-self.word_emb_dim:]])
        f.close()

        if self.char:
            # Char embeddings
            f = open(self.char_emb_path, 'r')

            for line in f:
                line = line.strip().split(' ')
                if len(line) > self.char_emb_dim + 1:
                    line[0] = ' '
                for key in self.replace.keys():
                    line[0] = line[0].replace(key, self.replace[key])
                self.char_emb[self.char_dict.lookup_strict(line[0])] = np.asarray([float(_) for _ in line[-self.char_emb_dim:]])
            f.close()

    def get_principal_components(self, x):
        # word level features
        ret1 = torch.zeros(self.r + 1, self.word_emb_dim).double()
        if len(''.join(x)) > 0:
            f = self.word_dict.lookup
            x_ = torch.from_numpy(np.array([self.word_emb[_] for _ in map(f, x)]))
            mu = torch.mean(x_, 0, keepdim=True)
            ret1[0,] = mu / torch.norm(mu)
            if self.r > 0:
                u, s, v = torch.svd(x_ - mu.repeat(x_.size(0), 1))
                k = self.r if v.size(1) > self.r else v.size(1)
                ret1[1:k+1,] = v.transpose(0, 1)[0:k, ]

        if self.char:
            # char level features
            ret2 = torch.zeros(self.r + 1, self.char_emb_dim).double()
            if len(''.join(x)) > 0:
                f = self.char_dict.lookup
                x_ = torch.from_numpy(np.array([self.char_emb[_] for _ in map(f, list(' '.join(x)))]))
                mu = torch.mean(x_, 0, keepdim=True)
                ret2[0,] = mu / torch.norm(mu)
                if self.r > 0:
                    u, s, v = torch.svd(x_ - mu.repeat(x_.size(0), 1))
                    k = self.r if v.size(1) > self.r else v.size(1)
                    ret2[1:k+1,] = v.transpose(0, 1)[0:k, ]

        if self.char:
            return torch.cat((ret1.view(1, -1), ret2.view(1, -1)), 1)
        else:
            return ret1.view(1, -1)

    def gen_feature(self, X):
        m1 = self.get_principal_components(X[1])
        m2 = self.get_principal_components(X[2])
        feature = torch.cat((m1, m2))
        if self.sent_feat:
            sent = self.get_principal_components(X[0])
            feature = torch.cat((feature, sent))
        if self.cont_feat:
            word_seq = self.get_principal_components(X[3])
            feature = torch.cat((feature, word_seq))

        feature = feature.view(1, -1)

        # add indicator feature for asymmetric relation
        if self.asymmetric:
            order = torch.zeros(1, 2).double()
            if X[4] == 0:
                order[0][0] = 1.0
            else:
                order[0][1] = 1.0
            feature = torch.cat((feature, order), 1).view(1, -1)

        return feature

    def _init_kwargs(self, **kwargs):

        self.model_kwargs = kwargs

        # Set if use char embeddings
        self.char = kwargs.get('char', True)

        # Set if use whole sentence feature
        self.sent_feat = kwargs.get('sent_feat', True)

        # Set if use whole context feature
        self.cont_feat = kwargs.get('cont_feat', True)

        # Set word embedding dimension
        self.word_emb_dim = kwargs.get('word_emb_dim', None)

        # Set word embedding path
        self.word_emb_path = kwargs.get('word_emb_path', None)

        # Set char embedding dimension
        self.char_emb_dim = kwargs.get('char_emb_dim', None)

        # Set char embedding path
        self.char_emb_path = kwargs.get('char_emb_path', None)

        # Set learning rate
        self.lr = kwargs.get('lr', 1e-3)

        # Set learning epoch
        self.n_epochs = kwargs.get('n_epochs', 100)

        # Set top k principal components
        self.r = kwargs.get('r', 10)

        # Set learning batch size
        self.batch_size = kwargs.get('batch_size', 100)

        # Set rebalance setting
        self.rebalance = kwargs.get('rebalance', False)

        # Set surrounding window size for mention
        self.window_size = kwargs.get('window_size', 3)

        # Set relation type indicator (e.g., symmetric or asymmetric)
        self.asymmetric = kwargs.get('asymmetric', False)

        # Set max sentence length
        self.max_sentence_length = kwargs.get('max_sentence_length', 100)

        # Replace placeholders in embedding files
        self.replace = kwargs.get('replace', {})

        print "==============================================="
        print "Number of learning epochs:     ", self.n_epochs
        print "Learning rate:                 ", self.lr
        print "Number of principal components:", self.r
        print "Batch size:                    ", self.batch_size
        print "Rebalance:                     ", self.rebalance
        print "Surrounding window size:       ", self.window_size
        print "Use sentence sequence          ", self.sent_feat
        print "Use window sequence            ", self.cont_feat
        print "Use char embeddings            ", self.char
        print "Word embedding size:           ", self.word_emb_dim
        print "Char embedding size:           ", self.char_emb_dim
        print "Word embedding:                ", self.word_emb_path
        print "Char embedding:                ", self.char_emb_path
        print "==============================================="

        assert self.word_emb_path is not None
        if self.char:
            assert self.char_emb_path is not None

    def train(self, X_train, Y_train, X_dev=None, Y_dev=None, print_freq=5, **kwargs):

        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """

        self._init_kwargs(**kwargs)

        # Set random seed
        torch.manual_seed(self.seed)

        # load embeddings from file
        self.load_embeddings()

        print "Done loading embeddings..."

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

        X_train= self._preprocess_data_combination(X_train, extend=False)
        if X_dev is not None:
            X_dev = self._preprocess_data_combination(X_dev, extend=False)

        Y_train = torch.from_numpy(Y_train).float()

        new_X_train = None
        for i in range(len(X_train)):
            feature = self.gen_feature(X_train[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X_train), feature.size(1))))
                print feature.size()
            new_X_train[i] = feature

        new_X_train = new_X_train.float()

        data_set = data_utils.TensorDataset(new_X_train, Y_train)
        train_loader = data_utils.DataLoader(data_set, batch_size=self.batch_size, shuffle=False)

        n_examples, n_features = new_X_train.size()
        n_classes = 1 if self.cardinality == 2 else None

        self.model = build_model(n_features, n_classes)
        loss = nn.MultiLabelSoftMarginLoss(size_average=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for idx in range(self.n_epochs):
            cost = 0.
            for x, y in train_loader:
                cost += train_model(self.model, loss, optimizer, x, y.float())
            if (idx + 1) % print_freq == 0:
                msg = "[%s] Epoch %s, Training error: %s" % (self.name, idx + 1, cost / n_examples)
                print msg

    def _marginals_batch(self, X):
        new_X_train = None
        X = self._preprocess_data_combination(X, extend=False)

        for i in range(len(X)):
            feature = self.gen_feature(X[i])
            if new_X_train is None:
                new_X_train = torch.from_numpy(np.zeros((len(X), feature.size(1))))
            new_X_train[i] = feature

        new_X_train = new_X_train.float()

        sigmoid = nn.Sigmoid()
        return sigmoid(predict(self.model, new_X_train)).data.numpy()

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
            if self.char:
                dump({'char_dict': self.char_dict, 'word_dict': self.word_dict, 'char_emb': self.char_emb,
                      'word_emb': self.word_emb}, f)
            else:
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
            if self.char:
                self.char_dict = d['char_dict']
                self.char_emb = d['char_emb']

        self.model = torch.load(os.path.join(model_dir, model_name))

        if verbose:
            print("[{0}] Loaded model <{1}>".format(self.name, model_name))
