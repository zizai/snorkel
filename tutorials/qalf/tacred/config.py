config = {
    'candidate_name' : 'TacredCandidate',
    'candidate_entities' : ['Subject', 'Object'],

    'splits': [0, 1],

    # Classifier
    'disc_model_class': 'logreg',
    'disc_params_range': {
        'n_epochs'  : [25, 50, 100],
    },
    'disc_params_default': {
        'lr':         0.01,
        'n_epochs':   100,
        'rebalance':  False,
        'batch_size': 64,
        'print_freq': 1,
    },
    'disc_eval_batch_size': 256,    
}