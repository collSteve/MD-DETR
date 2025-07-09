import numpy as np

def run_tsne(x: np.ndarray, **kwargs):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, **kwargs).fit_transform(x)