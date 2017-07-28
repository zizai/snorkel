import keras
import numpy as np
import os
import tensorflow as tf

from keras import backend as K
from six.moves.cPickle import dump, load
from snorkel.learning.classifier import Classifier
from snorkel.learning.utils import reshape_marginals, LabelBalancer


class KerasNoiseAwareModel(Classifier):
    """
    Generic NoiseAwareModel class for Keras models.
    Note that the actual network is built when train is called (to allow for
    model architectures which depend on the training data, e.g. vocab size).
    
    :param n_threads: Parallelism to use; single-threaded if None
    """
    def __init__(self, n_threads=None, **kwargs):
        self.n_threads = n_threads
        super(KerasNoiseAwareModel, self).__init__(**kwargs)

    def _build_model(self, **model_kwargs):
        """
        Builds the Keras model. Must set self.model.

        Note that _build_model is called in the train method, allowing for the 
        network to be constructed dynamically depending on the training set 
        (e.g., the size of the vocabulary for a text LSTM model)

        Note also that model_kwargs are saved to disk by the self.save method,
        as they are needed to rebuild / reload the model. *All hyperparameters
        needed to rebuild the model must be passed in here for model reloading
        to work!*
        """
        raise NotImplementedError()

    def _compile_model(self, opt, opt_kwargs, metrics=None):
        """Compiles Keras model"""
        # Define loss and marginals ops
        if self.cardinality > 2:
            loss_fn = keras.losses.categorical_crossentropy
        else:
            loss_fn = keras.losses.binary_crossentropy
        self.model.compile(opt(**opt_kwargs), loss_fn, metrics=metrics)

    def _build_new_graph_session(self, **model_kwargs):
        # Get model kwargs
        self.model_kwargs = model_kwargs
        # Create new session
        if self.n_threads is not None:
            K.set_session(tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=self.n_threads,
                inter_op_parallelism_threads=self.n_threads
            )))
        # Build model
        self._build_model(**model_kwargs)

    def _check_input(self, X):
        """Checks correctness of input; optional to implement."""
        pass

    def train(self, X_train, Y_train, n_epochs=25, opt=keras.optimizers.Adam,
        lr=0.01, batch_size=256, rebalance=False, X_dev=None, Y_dev=None,
        dev_ckpt=True, dev_ckpt_delay=0.75, callbacks=None,
        save_dir='checkpoints', verbose=True, **kwargs):
        """
        Generic training procedure for TF model

        :param X_train: Must be a NumPy matrix. Can be preprocessed via child.
        :param Y_train: Array of marginal probabilities for each Candidate
        :param n_epochs: Number of training epochs
        :param opt: Optimizer for model training
        :param lr: Learning rate
        :param batch_size: Batch size for SGD
        :param rebalance: Bool or fraction of positive examples for training
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        :param X_dev: Candidates for evaluation, same format as X_train
        :param Y_dev: Labels for evaluation, same format as Y_train
        :param dev_ckpt: If True, save a checkpoint whenever highest score
            on (X_dev, Y_dev) reached. Note: currently only evaluates at
            every @print_freq epochs.
        :param dev_ckpt_delay: Start dev checkpointing after this portion
            of n_epochs.
        :param callbacks: Keras callbacks
        :param save_dir: Save dir path for checkpointing.
        :param verbose: Be talkative?
        :param kwargs: All hyperparameters that change how the graph is built 
            must be passed through here to be saved and reloaded to save /
            reload model. *NOTE: If a parameter needed to build the 
            network and/or is needed at test time is not included here, the
            model will not be able to be reloaded!*
        """
        self._check_input(X_train)
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        # Check that the cardinality of the training marginals and model agree
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

        # Remove unlabeled examples (i.e. P(X=k) == 1 / cardinality for all k)
        # and optionally rebalance training set
        # Note: rebalancing only for binary setting currently
        if self.cardinality == 2:
            # This removes unlabeled examples and optionally rebalances
            train_idxs = LabelBalancer(Y_train).get_train_idxs(rebalance)
        else:
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            train_idxs = np.where(diffs > 1e-6)[0]
        # Input is either a NumPy array or a list of them for multi-input
        if isinstance(X_train, list):
            X_train = [X[train_idxs, :] for X in X_train]
        else:
            X_train = X_train[train_idxs, :]
        Y_train = Y_train[train_idxs]

        # Create new graph, build network, and start session
        self._build_new_graph_session(**kwargs)

        # Compile model
        opt_kwargs = {'lr': lr}
        self._compile_model(opt=opt, opt_kwargs=opt_kwargs)

        # Run mini-batch SGD
        dev_score_opt = 0.0
        n_delay = dev_ckpt_delay * n_epochs
        for t in range(n_epochs):
            # Run a partial fit
            self.model.fit(
                X_train, Y_train, batch_size=batch_size, epochs=t+1, verbose=1,
                callbacks=callbacks, shuffle=True, initial_epoch=t
            )
            # Check dev score
            if X_dev is not None:
                scores = self.score(X_dev, Y_dev)
                score = scores if self.cardinality > 2 else scores[-1]
                score_label = "Acc." if self.cardinality > 2 else "F1"
                print('Dev {0}={1:.2f}'.format(score_label, 100. * score))
                # If best score on dev set so far and dev checkpointing is
                # active, save checkpoint
                if dev_ckpt and t > n_delay and score > dev_score_opt:
                    dev_score_opt = score
                    self.save(save_dir=save_dir)

        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and dev_score_opt > 0:
            self.load(save_dir=save_dir)

    def save(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Save current model."""
        model_name = model_name or self.name

        # Note: Model checkpoints need to be saved in separate directories!
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_kwargs.pkl"), 'wb') as f:
            dump(self.model_kwargs, f)

        # Save graph and report if verbose
        self.model.save(model_dir + '.h5')
        if verbose:
            print("[{0}] Model saved as <{1}>".format(self.name, model_name))

    def load(self, model_name=None, save_dir='checkpoints', verbose=True):
        """Load model from file and rebuild in new graph / session."""
        model_name = model_name or self.name
        model_dir = os.path.join(save_dir, model_name)

        # Load model kwargs needed to rebuild model
        with open(os.path.join(model_dir, "model_kwargs.pkl"), 'rb') as f:
            model_kwargs = load(f)
        
        # Create new graph, build network, and start session
        self._build_new_graph_session(**model_kwargs)
        self.model = keras.models.load_model(model_dir + '.h5')

    def _preprocess_data(self, X):
        """Generic preprocessing subclass; may be called by external methods."""
        return X
