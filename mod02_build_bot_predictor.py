# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=250,
        max_depth=4,
        subsample=0.75,
        min_samples_leaf=15,
        random_state=314
    )
    model.fit(X, y)
    return model