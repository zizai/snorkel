import keras

from keras.layers import (
    concatenate, Bidirectional, Dense, Embedding, Input, LSTM, RepeatVector
)
from keras.models import Model

from .keras_disc_learning import KerasNoiseAwareModel


def arg_proxy(k):
    return "~~ARGUMENT_{0}~~".format(k)


def get_centered_subseq(seq, s, e, w, max_subseq_len=None):
    effective_l_w = s if s < w else w
    effective_r_w = len(seq) - 1 - e if (len(seq) - 1 - e < w) else w
    if max_subseq_len:
        while (e - s + 1 + effective_l_w + effective_r_w) > max_subseq_len:
            if effective_r_w > effective_l_w:
                effective_r_w -= 1
            else:
                effective_l_w -= 1
    return seq[max(0, s - effective_l_w) : e + 1 + effective_r_w]


class KerasMemNNExtractor(KerasNoiseAwareModel):

    def _process_candidate(self, c, arg_index_f, text_index_f,
        max_arg_len=None, max_side_len=None, max_btwn_len=None):
        w = self.window_size
        s = c.candidate.get_parent().words
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
            s_prox, l_s, l_e, w, max_side_len)))
        r_chunk = list(map(text_index_f, get_centered_subseq(
            s_prox, r_s, r_e, w, max_side_len)))
        # Get between window
        btwn_chunk = list(map(text_index_f, get_centered_subseq(
            s_prox, l_s, r_e, w, max_btwn_len)))
        # Return data
        return (arg_windows[0], arg_windows[1], l_chunk, r_chunk, btwn_chunk)        
    
    def _build_model(self, arg_len, side_len, btwn_len, arg_vocab_size,
        text_vocab_size, embedding_dim=100, rnn_hidden_dim=50, keep_prob=0.5,
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

        # Set the word dictionary passed in as the word_dict for the instance
        self.max_len = max_len
        self.word_dict = word_dict
        vocab_sz = word_dict.len()

        # Input argument windows
        arg1_input = Input(shape=(arg_len,), dtype='int32', name='arg1')
        arg2_input = Input(shape=(arg_len,), dtype='int32', name='arg2')

        # Embed argument windows
        arg_U = Embedding(output_dim=embedding_dim, input_dim=arg_vocab_size)
        arg1_vectors = arg_U(arg1_input)
        arg2_vectors = arg_U(arg2_input)
        arg1_embed = Bidirectional(cell_type(rnn_hidden_dim))(arg1_vectors)
        arg2_embed = Bidirectional(cell_type(rnn_hidden_dim))(arg2_vectors)

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
        l_concat = concatenate([l_vectors, side_arg1, side_arg2])
        r_concat = concatenate([r_vectors, side_arg1, side_arg2])
        btwn_concat = concatenate([btwn_vectors, btwn_arg1, btwn_arg2])

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
