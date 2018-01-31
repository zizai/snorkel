config = {
    'candidate_name' : 'TacredCandidate',
    'candidate_entities' : ['Subject', 'Object'],

    'splits': [0, 1],

    # Classifier
    'disc_model_class': 'logreg',
    'disc_params_range': {
        # 'lr'        : [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        # 'l1_penalty': [0, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
        # 'l2_penalty': [0, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4],
    },
    'disc_params_default': {
        'lr':         0.01,
        'l1_penalty': 0,
        'l2_penalty': 0,
        'n_epochs':   25,
        'rebalance':  False,
        'batch_size': 64,
        'print_freq': 1,
    },
    'disc_eval_batch_size': 256,    
}