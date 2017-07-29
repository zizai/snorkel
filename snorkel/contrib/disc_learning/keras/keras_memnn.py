import keras

from keras.layers import (
    add, concatenate, Bidirectional, Dense, Dropout,
    Embedding, Input, LSTM, RepeatVector
)
from keras.models import Model
from keras.preprocessing import sequence
from snorkel.learning.disc_models.rnn.utils import SymbolTable
from snorkel.models import Candidate
from snorkel.utils import ProgressBar

from .keras_disc_learning import KerasNoiseAwareModel


def arg_proxy(k):
    return "~~ARGUMENT_{0}~~".format(k)


def get_centered_subseq(seq, s, e, w, max_subseq_len=None, anchor=0):
    effective_l_w = s if s < w else w
    effective_r_w = len(seq) - 1 - e if (len(seq) - 1 - e < w) else w
    if max_subseq_len:
        while (e - s + 1 + effective_l_w + effective_r_w) > max_subseq_len:
            # Anchor left
            if anchor == 1:
                effective_r_w -= 1
            # Anchor right
            elif anchor == 2:
                effective_l_w -= 1
            # Centered
            else:
                if effective_r_w > effective_l_w:
                    effective_r_w -= 1
                else:
                    effective_l_w -= 1
    assert max(0, s - effective_l_w) < (e + 1 + effective_r_w)
    return seq[max(0, s - effective_l_w) : e + 1 + effective_r_w]


def scale_max_len(scale, *args):
    return int(scale * max(len(x) for y in args for x in y))


class KerasMemNNExtractor(KerasNoiseAwareModel):

    def _process_candidate(self, c, arg_index_f, text_index_f,
        max_arg_len=None, max_side_len=None, max_btwn_len=None):
        w = self.window_size
        s = c.get_parent().words
        # Get arg windows
        arg_windows = [list(map(arg_index_f, get_centered_subseq(
                s, c[k].get_word_start(), c[k].get_word_end(), w, max_arg_len
            ))) for k in [0, 1]
        ]
        # Figure out left and right
        l, r = (0,1) if c[0].get_word_start() < c[1].get_word_start() else (1,0)
        l_s, l_e = c[l].get_word_start(), c[l].get_word_end()
        r_s, r_e = c[r].get_word_start(), c[r].get_word_end()
        # Get proxied sentence
        s_prox = s[:]
        for k in [0, 1]:
            for i in range(c[k].get_word_start(), c[k].get_word_end() + 1):
                s_prox[i] = arg_proxy(k)
        # Get side windows
        l_chunk = list(map(text_index_f, get_centered_subseq(
            s_prox, 0, l_e, w, max_side_len, 2)))
        r_chunk = list(map(text_index_f, get_centered_subseq(
            s_prox, r_s, len(s)-1, w, max_side_len, 1)))
        # Get between window
        btwn_chunk = list(map(text_index_f, get_centered_subseq(
            s_prox, l_s, r_e, w, max_btwn_len)))
        # Return data
        return (arg_windows[0], arg_windows[1], l_chunk, r_chunk, btwn_chunk)

    def _process_candidates(self, candidates, extend=False, verbose=False):
        # Get lookup function
        if extend:
            arg_index_f = self.arg_word_table.get
            text_index_f = self.text_word_table.get
        else:
            arg_index_f = self.arg_word_table.lookup
            text_index_f = self.text_word_table.lookup
        # Set up progress bar
        if verbose:
            pb = ProgressBar(len(candidates))
        # Populate data
        X_arg1, X_arg2, X_l, X_r, X_btwn = [], [], [], [], []
        for i, candidate in enumerate(candidates):
            arg1, arg2, l, r, btwn = self._process_candidate(
                candidate, arg_index_f, text_index_f,
                self.max_arg_len, self.max_side_len, self.max_btwn_len
            )
            X_arg1.append(arg1)
            X_arg2.append(arg2)
            X_l.append(l)
            X_r.append(r)
            X_btwn.append(btwn)
            if verbose:
                pb.bar(i)
        if verbose:
            pb.close()
        return X_arg1, X_arg2, X_l, X_r, X_btwn

    def _pad_inputs(self, X):
        return [
            sequence.pad_sequences(X[0], maxlen=self.max_arg_len),
            sequence.pad_sequences(X[1], maxlen=self.max_arg_len),
            sequence.pad_sequences(X[2], maxlen=self.max_side_len),
            sequence.pad_sequences(X[3], maxlen=self.max_side_len),
            sequence.pad_sequences(X[4], maxlen=self.max_btwn_len),
        ]

    def train(self, X_train, Y_train, X_dev=None, window_size=2,
        max_arg_len=None, max_side_len=None, max_btwn_len=None, 
        len_scale=1.5, **kwargs):
        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """
        # Create word tables
        self.arg_word_table = SymbolTable()
        self.text_word_table = SymbolTable()
        # Save window size
        self.window_size = window_size
        # Set initial max lengths
        self.max_arg_len = max_arg_len
        self.max_side_len = max_side_len
        self.max_btwn_len = max_btwn_len
        # Preprocess training data
        print("Preprocessing training candidates")
        X_train = self._process_candidates(X_train, extend=True, verbose=True)
        # Update max lengths
        if max_arg_len is None:
            self.max_arg_len = scale_max_len(len_scale, X_train[0], X_train[1])
        if max_side_len is None:
            self.max_side_len = scale_max_len(len_scale, X_train[2], X_train[3])
        if max_btwn_len is None:
            self.max_btwn_len = scale_max_len(len_scale, X_train[4])
        print(self.max_arg_len, self.max_side_len, self.max_btwn_len)
        # Convert to padded matrices
        X_train = self._pad_inputs(X_train)
        # Process dev
        if X_dev is not None:
            X_dev = self._process_candidates(X_dev, extend=False)
            X_dev = self._pad_inputs(X_dev)
        # Train model, passing in things to save
        super(KerasMemNNExtractor, self).train(X_train, Y_train, X_dev=X_dev,
            window_size=self.window_size, arg_word_table=self.arg_word_table,
            text_word_table=self.text_word_table, max_arg_len=self.max_arg_len,
            max_side_len=self.max_side_len, max_btwn_len=self.max_btwn_len,
            **kwargs)
    
    def _build_model(self, embedding_dim=100, rnn_hidden_dim=50, keep_prob=0.5,
        mlp_n_hidden=1, mlp_hidden_dim=50, mlp_activation='relu', 
        cell_type=LSTM, word_dict=SymbolTable(), **kwargs):
        """
        Build RNN model
        
        :param dim: embedding dimension
        :param cell_type: RNN cell type
        :param batch_size: batch size for mini-batch SGD
        :param vocab_size: Vocab size
        """

        assert self.cardinality == 2
        if embedding_dim % 2 != 0:
            raise Exception("Embedding dimension must be even.")	

        arg_len = self.max_arg_len
        side_len = self.max_side_len
        btwn_len = self.max_btwn_len
        arg_vocab_size = self.arg_word_table.len()
        text_vocab_size = self.text_word_table.len()

        # Input argument windows
        arg1_input = Input(shape=(arg_len,), dtype='int32', name='arg1')
        arg2_input = Input(shape=(arg_len,), dtype='int32', name='arg2')

        # Embed argument windows in same dim as word vectors for add
        arg_U = Embedding(output_dim=embedding_dim, input_dim=arg_vocab_size)
        arg1_vectors = arg_U(arg1_input)
        arg2_vectors = arg_U(arg2_input)
        arg1_embed = Bidirectional(cell_type(embedding_dim//2))(arg1_vectors)
        arg2_embed = Bidirectional(cell_type(embedding_dim//2))(arg2_vectors)

        # Input sentence chunks
        l_input = Input(shape=(side_len,), dtype='int32', name='left_chunk')
        r_input = Input(shape=(side_len,), dtype='int32', name='right_chunk')
        btwn_input = Input(shape=(btwn_len,), dtype='int32', name='btwn_chunk')

        # Get sentence chunk vectors
        text_U = Embedding(output_dim=embedding_dim, input_dim=text_vocab_size)
        l_vectors = text_U(l_input)
        r_vectors = text_U(r_input)
        btwn_vectors = text_U(btwn_input)

        # Combine sentence chunk vectors and argument embeddings
        side_arg1 = RepeatVector(side_len)(arg1_embed)
        side_arg2 = RepeatVector(side_len)(arg2_embed)
        btwn_arg1 = RepeatVector(btwn_len)(arg1_embed)
        btwn_arg2 = RepeatVector(btwn_len)(arg2_embed)
        l_concat = add([l_vectors, side_arg1, side_arg2])
        r_concat = add([r_vectors, side_arg1, side_arg2])
        btwn_concat = add([btwn_vectors, btwn_arg1, btwn_arg2])

        # Embed sentence chunks and merge
        l_embed = Bidirectional(cell_type(rnn_hidden_dim))(l_concat)
        r_embed = Bidirectional(cell_type(rnn_hidden_dim))(r_concat)
        btwn_embed = Bidirectional(cell_type(rnn_hidden_dim))(btwn_concat)
        embedded_sentence = concatenate([l_embed, r_embed, btwn_embed])

        # Add dropout
        h = Dropout(1. - keep_prob)(embedded_sentence)

        # Add MLP
        for _ in range(mlp_n_hidden):
            h = Dense(mlp_hidden_dim, activation=mlp_activation)(h)
            h = Dropout(1. - keep_prob)(h)
        
        # Prediction layer
        predictions = Dense(1, activation='sigmoid')(h)

        # Construct model
        self.model = Model(
            inputs=[arg1_input, arg2_input, l_input, r_input, btwn_input],
            outputs=predictions
        )

    def marginals(self, test_candidates):
        """Get likelihood of tagged sequences represented by test_candidates
            @test_candidates: list of lists representing test sentence
        """
        # Preprocess if not already preprocessed
        if isinstance(test_candidates[0], Candidate):
            X_test = self._process_candidates(test_candidates, extend=False)
            X_test = self._pad_inputs(X_test)
        else:
            X_test = test_candidates
        # Run feed-forward
        return self.model.predict(X_test, batch_size=256)
