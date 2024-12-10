# config.py
ENTRY_NUMBER_LAST_DIGIT = 9 # change with yours
ENTRY_NUMBER = '2021MT60949'

PRE_PROCESSING_CONFIG ={
    "hard_margin_linear" : {
        "use_pca" : None,
    },

    "hard_margin_rbf" : {
        "use_pca" : 10,
    },

    "soft_margin_linear" : {
        "use_pca" : 10,
    },

    "soft_margin_rbf" : {
        "use_pca" : 10,
    },

    "AdaBoost" : {
        "use_pca" : 50,
    },

    "RandomForest" : {
        "use_pca" : 50,
    }
}

SVM_CONFIG = {
    "hard_margin_linear": {
        "C": 1e9,  # A large value to simulate hard margin
        "kernel": 'linear',
        "val_score": 0.9923,  # Replace with your expected validation score
    },
    "soft_margin_linear": {
        "C": 0.1,
        "kernel": 'linear',
        "val_score": 0.9871,  # Replace with your expected validation score
    },
    "hard_margin_rbf": {
        "C": 1e9,  # A large value to simulate hard margin
        "kernel": 'rbf',
        "gamma": 0.01,  # Adjust gamma based on your data
        "val_score": 0.9848,  # Replace with your expected validation score
    },
    "soft_margin_rbf": {
        "C": 0.1,
        "kernel": 'rbf',
        "gamma": 0.01,  # Adjust gamma based on your data
        "val_score": 0.9714,  # Replace with your expected validation score
    }
}
ENSEMBLING_CONFIG = {
    'AdaBoost':{
        'num_trees' : 50,
        "val_score" : 0.9854,
    },

    'RandomForest': {
        'num_trees': 20,
        'max_depth': 10,
        'min_samples_split': 5,
        'max_features': None,
        "val_score": 0.9727,  # Replace with your validation score after tuning
    }

}


