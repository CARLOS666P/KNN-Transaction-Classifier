
from sklearn.decomposition import PCA

def pcanorm_reduc(x,y):
    """Load a DataFrame and chose the n component.
    Args:
    x: A data frame only with values and no labels 
    y:qty of commponent that you keep
    Return:
    A numpy.ndarray with the n components  """
    pca = PCA(n_components=y)
    pca.fit(x)
    return pca.transform(x)