config = {
    'candidate_name' : 'TacredCandidate',
    'candidate_entities' : ['Subject', 'Object'],

    # Classifier
    'disc_model_class': 'lstm',
    'disc_model_search_space': 10,
	'disc_init_params': {
        'n_threads': 16,
        'seed'     : None,
    },
    'disc_params_range': {
        'lr'        : [1e-2, 1e-3, 1e-4],
        'n_epochs'  : [10, 20, 50],
        'batch_size': [32, 64, 128, 256],
    },
    'disc_params_default': {
        'lr':         0.01,
        'dim':        50,
        'n_epochs':   20,
        'dropout':    0.5,
        'rebalance':  0.25,
        'batch_size': 128,
        'max_sentence_length': 200,
        'print_freq': 1,
    },
    'disc_eval_batch_size': 256,    
}