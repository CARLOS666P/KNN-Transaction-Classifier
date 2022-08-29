import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.externals 
import joblib

#load data
df = pd.read_excel("datatrans.xlsx")


#Distribution of labels

inv=df[df["Resolution"]==1]
fp=df[df["Resolution"]==0]
all=df.shape[0]

x = len(inv)/all
y = len(fp)/all

print('inv :',x*100,'%')
print('fp :',y*100,'%')

labels = ['false positive','investigation case']
classes = pd.value_counts(df['Resolution'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Resolution distribution")
plt.xticks(range(2), labels)
plt.xlabel("Resolution")
plt.ylabel("Frequency")

plt.show()

#data without labels

dfnl=df.drop(["Resolution"],axis=1)

#import function  to plot the PCA var from components

import functions.pcavar as pcavar 

pcavar.pcavar_plot(dfnl)


#after see the last plot you can reduce the principal coponents with the following function

import functions.pcastd_reduc as pcastd_reduc

pcaft=pcastd_reduc.pcanorm_reduc(dfnl,3)

#taking the variable to train and test the model

X = pd.DataFrame(pcaft,columns=["PC1","PC2","PC3"])
y = df["Resolution"]

import functions.KN_test as KNF

KNF.K_testplot(X,y,10,0.15)

#creating a new balance dataset for test the model

df = df.sample(frac=1)

Investigation = df[df['Resolution'] == 1]
False_postive = df[df['Resolution'] == 0][:180]

new_df = pd.concat([False_postive, Investigation])
# Shuffle dataframe rows
new_df = new_df.sample(frac=1, random_state=42)

new_df.head()

# Let's plot the Transaction class against the Frequency
labels = ['False positive','Investigation']
classes = pd.value_counts(new_df['Resolution'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("sample test set resolution distribution")
plt.xticks(range(2), labels)
plt.xlabel("Resolution")
plt.ylabel("Frequency")
plt.show()

# prepare the data
features = new_df.drop(['Resolution'], axis = 1)
labels = pd.DataFrame(new_df['Resolution'])

feature_array = features.values
label_array = labels.values

# splitting the faeture array and label array keeping 85% for the trainnig sets
X_train,X_test,y_train,y_test = train_test_split(feature_array,label_array,test_size=0.15)



# Create a KNN classifier with 5 neighbors

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train.ravel())

#saving the train model

filename = 'finalized_model.sav'
joblib.dump(knn, filename)

knn = joblib.load(filename)

#making prediction on the test set

knnpredict=knn.predict(X_test)

#accuraacy test and confusion matrix

import functions.accuracytestCM as acc
  
acc.accuracy_test(y_test,knnpredict)