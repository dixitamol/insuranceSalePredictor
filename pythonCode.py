#Importing package
import numpy as n
import matplotlib.pyplot as pl
import pandas as p
import seaborn as s
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
var=16 
print('Variables selected :  ', list(df.columns.values[[10,16,25,29,31,33,40,41,42,47,59,61,68]]))
selected = df.columns.values[[10,16,25,29,31,33,40,41,42,47,59,61,68]]
X = (df[df.columns[[10,16,25,29,31,33,40,41,42,47,59,61,68]]].values)
# Normalization - Using MinMax Scaler
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

y = n.vstack(df['CARAVAN'].values)

print('\n')
print('X and y Input Data:   ', X.shape, y.shape)


X_train_original, X_test2, y_train_original, y_test2 = train_test_split(X, y, test_size=0.3,
                                                                        random_state=42)

print('Training Set Shape:   ', X_train_original.shape, y_train_original.shape)

X_val, X_test, y_val, y_test = train_test_split(X_test2, y_test2, test_size=0.33,random_state=42)
# Used Seed in Partitioning so that Test Set remains same for every Run

print('Validation Set Shape: ', X_val.shape,y_val.shape)
print('Test Set Shape:       ', X_test.shape, y_test.shape)



#LOADING DATASET
df = p.read_csv("dat234.csv")


#BAR PLOT OF CARAVAN DETAILS:
no=sum(df['CARAVAN']==0)
yes=sum(df['CARAVAN']==1)
colors=['red','blue']
locations=[1,2]
heights=[no,yes]
labels=['NO','Yes']
pl.bar(locations,heights,color=colors,tick_label=labels,alpha=0.7)
pl.xlabel('CARAVAN')
pl.ylabel('Customer Subtype')
pl.title('Caravan Details')


#POLICY PERCENTILE:
def PropByVar(df, variable):
    dataframe_pie = df[variable].value_counts()
    ax = dataframe_pie.plot.pie(figsize=(10,10), autopct='%1.2f%%', fontsize = 12);
    ax.set_title(variable + ' (%) Per Customer Subtype \n', fontsize = 15);
    return n.round(dataframe_pie/df.shape[0]*100,2)
PropByVar(df, 'CARAVAN')


#COMPARE ACCURACY OF MODELS ON VALIDATION SET:
print('       Accuracy of Models       ')
print('--------------------------------')
print('Decision Tree           '+"{:.2f}".format(accuracy_score(y_val, y_pred_DT)*100)+'%')
print('Neural Network          '+"{:.2f}".format(accuracy_score(y_val, y_pred_MLP)*100)+'%')
print('Logistic Regression     '+"{:.2f}".format(accuracy_score(y_val, y_pred_Log)*100)+'%')
print('Random Forest           '+"{:.2f}".format(accuracy_score(y_val, y_pred_RF)*100)+'%')
print('Support Vector Machine  '+"{:.2f}".format(accuracy_score(y_val, y_pred_SVM)*100)+'%')


#ConfusionMatrix
from sklearn.metrics import confusion_matrix

print('Decision Tree  ')
cm_DT = confusion_matrix(y_val, y_pred_DT)
print(cm_DT)
print('\n')
print('Neural Network  ')
cm_DT = confusion_matrix(y_val, y_pred_MLP)
print(cm_DT)
print('\n')
print('Logistic Regression  ')
cm_DT = confusion_matrix(y_val, y_pred_Log)
print(cm_DT)
print('\n')
print('Random Forest  ')
cm_DT = confusion_matrix(y_val, y_pred_RF)
print(cm_DT)
print('\n')
print('Support Vector Machine  ')
cm_DT = confusion_matrix(y_val, y_pred_SVM)
print(cm_DT)
print('\n')



#DATA VISUALIZATION HEAT MAP:
s.heatmap(df.corr())
