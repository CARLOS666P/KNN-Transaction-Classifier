
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report
import seaborn as sns
from matplotlib import pyplot as plt

def accuracy_test(ytest,ypred):
 
    """Load the test and predict variables.
    Args:
    yteststr: y_test from train test split
    ypred: model.predict with a test set 

    return accuracy test an a plot with confusion matrix
    """
    knn_accuracy_score  = accuracy_score(ytest,ypred)
    knn_precison_score  = precision_score(ytest,ypred)
    knn_recall_score    = recall_score(ytest,ypred)
    knn_f1_score        = f1_score(ytest,ypred)
    knn_MCC             = matthews_corrcoef(ytest,ypred)

    LABELS = ['False Positive', 'Investigation case']
    conf_matrix = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class') 
    plt.show()   
    print("MCC -->",knn_MCC)
    

    return print(classification_report(ytest,ypred))
