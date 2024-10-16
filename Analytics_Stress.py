import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#ML
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from pandas.core.common import random_state
#ML Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
data = pd.read_csv("C:/Users/PRIYA/Desktop/code_Human_stress/stress.csv")
print("",data.head())
     
print("",data.info())

data.describe()
print("",data.shape)

#initializing Colums for future use
data.columns = ['snoring_rate', 'respiration_rate', 'body_temperature', 
'limb_movement', 'blood_oxygen','eye_movement', 'sleeping_hours', 'heart_rate',
 'stress_level']

# Finding Null
data.isnull().sum()

plt.figure(figsize=(20,5))
sns.lineplot(x='snoring_rate',y='stress_level',data=data)
plt.xlabel("Snoring Rate")
plt.ylabel('Stress Level')
plt.title('Snoring Rate vs Stress Level')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(20,5))
sns.lineplot(x='respiration_rate',y='stress_level',data=data)
plt.xlabel("Respiration Rate")
plt.ylabel('Stress Level')
plt.title('Respiration Rate vs Stress Level')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(20,5))
sns.lineplot(x='body_temperature',y='stress_level',data=data)
plt.xlabel("Body Temperature")
plt.ylabel('Stress Level')
plt.title('Body Temperature vs Stress Level')
plt.xticks(rotation=0)
plt.show()


plt.figure(figsize=(20,5))
sns.scatterplot(x='blood_oxygen',y='stress_level',data=data)
plt.xlabel("Blood Oxygen")
plt.ylabel('Stress Level')
plt.title('Blood Oxygen vs Stress Level')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(20,5))
sns.scatterplot(x='eye_movement',y='stress_level',data=data)
plt.xlabel("Eye Movement")
plt.ylabel('Stress Level')
plt.title('Eye Movement vs Stress Level')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(20,5))
sns.scatterplot(x='heart_rate',y='stress_level',data=data)
plt.xlabel("Heart Rate")
plt.ylabel('Stress Level')
plt.title('Heart Rate vs Stress Level')
plt.xticks(rotation=0)
plt.show()  

data.columns = ['snoring_rate', 'respiration_rate', 'temperature', 'limb_move', 'blood_oxygen', 'eye_move', 'sleep_hour', 'heart_rate', 'stress_level']
color = '#eab889'
data = data.copy()

data.hist(bins=15,figsize=(25,15),color=color)
plt.rcParams['font.size'] = 18
plt.show()

plt.figure(figsize=(10,10))
tc = data.corr()
sns.heatmap(tc,annot=True)
plt.title('Heatmap')
plt.show()

data.plot(kind='box', subplots=True, layout=(2,14),figsize=(14,14), sharex=False, sharey=False)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=2, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()
     

x = data.copy();
x.drop('stress_level', axis = 1, inplace = True)
y = data['stress_level']

#Normalize data features using minmax.
x = minmax_scale(x)
#Separating training data and testing data with a ratio of 80%:20%.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2,random_state =123)
data_preprocessed = pd.DataFrame(x, columns = ['snoring_rate', 'respiration_rate', 'temperature', 'limb_move', 'blood_oxygen', 'eye_move', 'sleep_hour', 'heart_rate'])
data_preprocessed['stress_level'] = y
data_preprocessed.head()

k = [5,10,15]
max_depth = [20, 40, 60]
for i in range(3):
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors=k[i])))
    models.append(('NB', GaussianNB()))
    models.append(('DT', DecisionTreeClassifier(max_depth=max_depth[i], random_state=101)))
    results = []
    names = []
    print("K:",k[i], "and max_depth:", max_depth[i])
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    print("")
    
    
from sklearn.metrics import accuracy_score

# Create K-NN Model
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train, y_train)
predict_KNN = model_KNN.predict(X_test)
# Create NB Model
model_NB = GaussianNB()
model_NB.fit(X_train, y_train)
predict_NB = model_NB.predict(X_test)
# Create DT Model
model_DT = DecisionTreeClassifier(max_depth=40,random_state=101)
model_DT.fit(X_train, y_train)
predict_DT = model_DT.predict(X_test)


confusion_matrix = metrics.confusion_matrix(y_test, predict_KNN)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4])
cm_display.plot()
plt.show()

print(classification_report(y_test, predict_KNN))

#NB
confusion_matrix = metrics.confusion_matrix(y_test, predict_NB)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4])

cm_display.plot()
plt.show()
y_t = y_test
#DT
confusion_matrix = metrics.confusion_matrix(y_test, predict_DT)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4])
cm_display.plot()
plt.show()

print(classification_report(y_test, predict_DT))


data = pd.read_csv("C:/Users/PRIYA/Desktop/code_Human_stress/stress.csv")
data.rename(columns = {'sr':'snoring rate', 'rr':'respiration rate',
                        't':'body temperature', 'lm':'limb movement', 
                        'bo':'blood oxygen', 'rem':'eye movement', 
                        'sr.1':'sleeping hours','hr':'heart rate', 
                        'sl':'stress level'}, inplace = True)
data.head()
#Stress Levels (0- low/normal, 1 – medium low, 2- medium, 3-medium high, 4 -high)

from sklearn.preprocessing import MinMaxScaler
#Defining varible
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(data[['snoring rate', 'respiration rate',
'body temperature', 'limb movement','blood oxygen', 'eye movement', 
'sleeping hours','heart rate']])
print(scaled)
newdf = pd.DataFrame(scaled, columns =['snoring rate', 'respiration rate', 'body temperature', 'limb movement',
       'blood oxygen', 'eye movement', 'sleeping hours', 'heart rate'])
newdf.head()
newdf['stress level']=data['stress level']


newdf
newdf.describe()
newdf['stress level'].value_counts().sort_values()

#plotting a pie chart to show the distribution of data
label = ['low/normal' , 'medium low' , 'medium' ,'medium low','high']
ex=[0.1,0,0,0,0.1]
plt.pie(newdf['stress level'].value_counts(),labels=label,autopct='%.0f%%',explode=ex,shadow=True)
plt.show()
newdf.corrwith(newdf['stress level'], method = 'pearson')
plt.figure(figsize=(10,6))
sns.heatmap(newdf.corr(),annot=True)

#splitting among features and target
from sklearn.model_selection import train_test_split as tts
X = newdf[['snoring rate', 'respiration rate', 'body temperature', 'limb movement',
       'blood oxygen', 'eye movement', 'sleeping hours', 'heart rate']]
y = newdf['stress level']
#splitting among test and train dataset
x_train, x_test, y_train, y_test= tts(X, y, test_size=0.4)
print('Dimensions of train dataset:',x_train.shape)
print('Dimensions of test dataset:',x_test.shape)

#defining dictionaries for storing results of different models and comparing 
sc = {}
rn = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score as cvs
lrr=LogisticRegression()
lrr.fit(x_train,y_train)
lrr_pred=lrr.predict(x_test)
print('accuracy_score:',metrics.accuracy_score(y_test, lrr_pred))

r=cvs(lrr, X, y, cv=10, scoring='accuracy').mean()
sc['Logistic Regression']=r
rn['Logistic Regression']=np.array(np.unique(lrr_pred, return_counts=True))
print('cross val score:',r)

#XGB Classifier
from xgboost import XGBClassifier
# Corrected version
XGB = XGBClassifier()
# Fit the model without explicitly setting 'eval_metric' in fit()
XGB.fit(x_train, y_train, 
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=True)
xgb_pred=XGB.predict(x_test)
print("accuracy_score",metrics.accuracy_score(y_test, xgb_pred))
sc['XGB Classifier']=metrics.accuracy_score(y_test, xgb_pred)
rn['XGB Classifier']=np.array(np.unique(xgb_pred, return_counts=True))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier(n_estimators = 50)
rfm.fit(x_train,y_train)
rfm_pred=rfm.predict(x_test)
print("accuracy_score",metrics.accuracy_score(y_test, rfm_pred))
sc['Random Forest']=metrics.accuracy_score(y_test, rfm_pred)
rn['Random Forest']=np.array(np.unique(rfm_pred, return_counts=True))

print("",sc)
print("",rn)

import matplotlib.pyplot as plt

classifiers = ['Logistic Regression', 'XGB Classifier', 
               'Random Forest Classifier', 'Navie Bayes', 
               'Decesion Tree']
accuracies = [metrics.accuracy_score(y_test, lrr_pred),
              metrics.accuracy_score(y_test, xgb_pred) ,
              metrics.accuracy_score(y_test, rfm_pred),
              accuracy_score(y_t, predict_NB), 
              accuracy_score(y_t, predict_DT)]

# plt.bar(classifiers, accuracies, color = 'blue')
# plt.xlabel('Classifiers')
# plt.ylabel('Accuracies')
# plt.title('Accuracy of Classifiers')
# plt.legend(loc='upper left')
# plt.show()

fig, ax = plt.subplots(figsize=(30, 15))

x_pos = np.arange(len(classifiers))

ax.bar(x_pos, accuracies, align='center', color=['red', 'green', 'blue'])
ax.set_xticks(x_pos)
ax.set_xticklabels(classifiers)
ax.set_ylim(0, 1)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Classifiers')

for i, v in enumerate(accuracies):
    ax.text(i, v + 0.05, str(v), color='black', fontweight='bold')

ax.legend(['Accuracy'], loc='upper left')
plt.show()
     