import keras
import warnings

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing import sequence
from snorkel.models import Candidate

from .keras_disc_learning import KerasNoiseAwareModel

from snorkel.learning.disc_models.rnn.utils import SymbolTable
from snorkel.learning.disc_models.rnn import reRNN, TagRNN, TextRNN


class KerasRNNBase(KerasNoiseAwareModel):
    
    representation = True

    def _preprocess_data(self, candidates, extend):
        """Build @self.word_dict to encode and process data for extraction
            Return list of encoded sentences, list of last index of arguments,
            and the word dictionary (extended if extend=True)
        """
        raise NotImplementedError()

    def _check_max_sentence_length(self, ends, max_len=None):
        """Check that extraction arguments are within @self.max_len"""
        mx = max_len or self.max_len
        for i, end in enumerate(ends):
            if end >= mx:
                w = "Candidate {0} has argument past max length for model:"
                info = "[arg ends at index {0}; max len {1}]".format(end, mx)
                warnings.warn('\t'.join([w.format(i), info]))

    def _build_model(self, embedding_dim=100, hidden_dim=50, max_len=20,
        dropout=0.5, cell_type=LSTM, word_dict=SymbolTable(), **kwargs):
        """
        Build RNN model
        
        :param dim: embedding dimension
        :param cell_type: RNN cell type
        :param batch_size: batch size for mini-batch SGD
        :param vocab_size: Vocab size
        """

        # Set the word dictionary passed in as the word_dict for the instance
        self.max_len = max_len
        self.word_dict = word_dict
        vocab_sz = word_dict.len()

        self.model = Sequential()
        self.model.add(Embedding(vocab_sz, embedding_dim, input_length=max_len))
        self.model.add(Bidirectional(cell_type(hidden_dim)))
        self.model.add(Dropout(dropout)) 
        
        # Build activation layer
        if self.cardinality > 2:
            self.model.add(Dense(self.cardinality, activation='softmax'))
        else:
            self.model.add(Dense(1, activation='sigmoid'))

    def train(self, X_train, Y_train, X_dev=None, max_sentence_length=None, 
        max_sentence_length_scale=2, **kwargs):
        """
        Perform preprocessing of data, construct dataset-specific model, then
        train.
        """
        # Text preprocessing
        X_train, ends = self._preprocess_data(X_train, extend=True)
        if X_dev is not None:
            X_dev, _ = self._preprocess_data(X_dev, extend=False)
        
        # Get max sentence size
        max_len = max_sentence_length or 2 * max(len(x) for x in X_train)
        self._check_max_sentence_length(ends, max_len=max_len)
        
        # Convert to arrays
        X_train = sequence.pad_sequences(X_train, maxlen=max_len)
        X_dev = sequence.pad_sequences(X_dev, maxlen=max_len)

        # Train model- note we pass word_dict through here so it gets saved...
        super(KerasRNNBase, self).train(X_train, Y_train, X_dev=X_dev,
            word_dict=self.word_dict, max_len=max_len, **kwargs)

    def marginals(self, test_candidates):
        """Get likelihood of tagged sequences represented by test_candidates
            @test_candidates: list of lists representing test sentence
        """
        # Preprocess if not already preprocessed
        if isinstance(test_candidates[0], Candidate):
            X_test, ends = self._preprocess_data(test_candidates, extend=False)
            self._check_max_sentence_length(ends)
            X_test = sequence.pad_sequences(X_test, maxlen=self.max_len)
        else:
            X_test = test_candidates

        return self.model.predict(X_test, batch_size=256)


class KerasreRNN(KerasRNNBase):
    def _preprocess_data(self, candidates, extend=False):
        return reRNN._preprocess_data(self, candidates, extend=extend)


class KerasTagRNN(KerasRNNBase):
    def _preprocess_data(self, candidates, extend=False):
        return TagRNN._preprocess_data(self, candidates, extend=extend)


class KerasTextRNN(KerasRNNBase):
    def _preprocess_data(self, candidates, extend=False):
        return TextRNN._preprocess_data(self, candidates, extend=extend)
