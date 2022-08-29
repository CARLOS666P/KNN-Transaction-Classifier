from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt

def pcavar_plot(path):
   """Load a DataFrame.
   Args:
   path(str): A data frame only with values and no labels 
   Return:
      A plot with the explained variance of the princial components   
   """
   scaler = StandardScaler()
   pca1 = PCA() 
   pipeline = make_pipeline( scaler ,pca1)
   pipeline.fit(path)
   features = range(pca1.n_components_)
   plt.bar(features, pca1.explained_variance_)
   plt.xlabel('PCA feature')
   plt.ylabel('variance')
   plt.xticks(features) 
   return plt.show()