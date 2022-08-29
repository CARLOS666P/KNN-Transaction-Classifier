
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np 

def K_testplot(X,y,n,alpha):
 """Load the variables to split the data and test the accuracy of the model.
    Args:
    X: Components to predict labels 
    y:labels to predict
    n: max qty of k neighbors to test
    alpha: percent of the split data to the test set
    Return:
 A plot with the accuracy of the test and train set   
    """
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=alpha, random_state=42, stratify=y)
 neighbors = np.arange(1,n)
 train_accuracies = {}
 test_accuracies = {}
 for neighbor in neighbors:
    
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
 plt.title("KNN: Varying Number of Neighbors")
 plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
 plt.plot(neighbors,test_accuracies.values(), label="Testing Accuracy")
 plt.legend()
 plt.xlabel("Number of Neighbors")
 plt.ylabel("Accuracy")
 return plt.show()