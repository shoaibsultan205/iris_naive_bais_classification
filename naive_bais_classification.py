import numpy as np 
import pandas as pp 
import matplotlib.pyplot as ppp
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

dataset=pp.read_csv('C:/Users/PC/Downloads/Iris.csv')
#data preprocessing 
#apply label encoding 

label=LabelEncoder()
dataset['Species']=label.fit_transform(dataset['Species'])

x=dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

sts=StandardScaler()
x_train=sts.fit_transform(x_train)
x_test=sts.fit_transform(x_test)

#apply naive baise theoram 
# it's basically use for continuous value 
gau=GaussianNB()
gau.fit(x_train,y_train)


pre_y=gau.predict(x_test)

from sklearn.metrics import recall_score,confusion_matrix,f1_score,roc_auc_score,accuracy_score
#recall_score(y_test,y_pre)
confusion_matrix(y_test,pre_y)
accuracy_score(y_test,pre_y)

#visualization 
# ✅ PCA Visualization: Correct vs Incorrect Predictions
pca = PCA(n_components=2)
x_test_2d = pca.fit_transform(x_test)

# Identify correct/incorrect predictions
correct = (pre_y == y_test.values)
colors = ['green' if c else 'red' for c in correct]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1], c=colors, s=100, edgecolors='k')

# Add legend
correct_patch = mpatches.Patch(color='green', label='Correct Prediction')
incorrect_patch = mpatches.Patch(color='red', label='Incorrect Prediction')
plt.legend(handles=[correct_patch, incorrect_patch], title='Prediction Result')

# Labels and title
plt.title('Naive Bayes: Correct vs Incorrect Predictions (PCA Projection)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
