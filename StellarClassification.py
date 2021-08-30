# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:25:53 2021

@author: veyse
"""

import pandas as pd #to load and manipulate data for One-Hot Encoding
import numpy as np #to calculate the mean and standard deviation
import matplotlib.pyplot as plt #to draw graphs
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
#Importing Modules
from sklearn.tree import DecisionTreeClassifier #build a classification tree
from sklearn.tree import plot_tree #to draw a classification tree
from sklearn.model_selection import train_test_split #to split data into training and testing sets
from sklearn.model_selection import cross_val_score #for cross validation
from sklearn.metrics import confusion_matrix#to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix#to draw a confusion matrix
import timeit



# ======================= DATA READING ======================

df = pd.read_csv('Stars.csv')

print(df.to_string()) 



#Add header to the dataframe
#names = ['temperature', 'L',....]
#df = pd.read_csv('Stars.csv',header=None, names = names)


# ======================= DATA READING ======================



# ======================= SCATTER PLOTS =====================


#Convert categorical values into numeric values
le = LabelEncoder()
df['Color'] = le.fit_transform(df['Color'])
df['Spectral_Class'] = le.fit_transform(df['Spectral_Class'])



df.plot.scatter(x = 'Temperature', y = 'L')


sns.set_style('white')
sns.set_style('ticks')
sns.regplot(x = 'L', y = 'R', color = 'g', data = df)


sns.lmplot(x = 'Spectral_Class', y ='A_M', hue = 'Type', data = df)


sns.pairplot(df, hue = 'Type', palette = 'Set1')
# ======================= SCATTER PLOTS ======================







# ======================= DATA DESCRIBING ====================

#Get row and column numbers
print(df.shape)

#Get column names, NA values, count, and types
print(df.info())

#General statistical descriptions of columns
print(df.describe())

#Get names of the columns
print(df.columns)

#Get types of the columns
print(df.dtypes)

#Print different values of the column
print(df['Color'].unique())

# ======================= DATA DESCRIBING ====================

# ======================= DATA FORMATING =====================

#No mistakes in the Spectral Class column
print(df['Spectral_Class'].unique())

#There are some mistakes in the Color column
print(df['Color'].unique())
print(df['Color'].value_counts())

df['Color'] = df['Color'].str.lower()

#df.loc takes two arguments, 'row index' and 'column index'. We are checking if the value is 'blue white' of each
#row value, under 'Color' column and then we replacing it with 'blue-white'.
df.loc[df['Color'] == 'blue white','Color'] = 'blue-white'
df.loc[df['Color'] == 'white-yellow','Color'] = 'yellow-white'

#The capitalize() function returns a string with the first character in the capital. 
df['Color'] = df['Color'].str.capitalize()


# Blue-white  26              Blue White  10       Blue white 4
#
#
#

# ======================= DATA FORMATING =====================



# ======================= DATA SLICING =======================

#Get row 1 and 20 and temperature and color column values
print(df.loc[[1, 20],['Temperature','Color']])

#Get rows 1, 3, 8, 20 and first & second column values
print(df.iloc[[1,3,8,20],[0, 1]])

#Get rows 0 and 1 with all column values
print(df.iloc[[0,1]])

#Get all rows with only temperature and color column values
print(df[['Temperature','Color']])

#If color is red and temperature is smaller than 4000
print(df[(df['Color'] == 'Red') & (df['Temperature'] <= 4000)])

# ======================= DATA SLICING =======================


# ======================= DATA REMOVING ======================

##Drop row and columns
df.drop(['Color', 'Type'], axis = 1, inplace = True) # Column
df.drop([1,2,3,4,5], axis = 0) # Rows

# ======================= DATA REMOVING ======================


# ======================= MISSING DATA =======================

#Count the number of rows that have missing variable
#loc[], short for 'location', let's us specify which rows we want ...
#and so we say we want any row with '?' in column 'ca'
#OR
#any row with '?' in the column 'thal'
#
#len(), short for 'length' prints out the number of rows.
len(df.loc[(df['Color']) == '?'] | (df['Spectral_Class']) == '?')

#print out the rows that contain missing values.
print(df.loc[(df['Color']) == '?'] | (df['Spectral_Class']) == '?')

print(len(df))

#If rows with missing values are so few than delete the rows
# ======================= MISSING DATA =======================


# ======================= DATA COUNTING ======================

# Counts of categorical outcomes
print(df['Color'].value_counts())
print(df['Spectral_Class'].value_counts())
print(df['Type'].value_counts())

print(df['Temperature'].value_counts())

# ======================= DATA COUNTING ======================


# ======================= DATA GROUPING ======================

#Groupby
df.groupby('Color')['Spectral_Class'].value_counts()

# ======================= DATA GROUPING ======================


# ======================= DUPLICATED ROWS ====================

#Duplicated Rows
df.duplicated()
df.duplicated(['Color'])

# ======================= DUPLICATED ROWS ====================

# ======================= HISTOGRAMS =========================

# Continious Values Outlier
plt.hist(df['Temperature'])
plt.show()


plt.hist(df['L'])
plt.show()


plt.hist(df['R'])
plt.show()


plt.hist(df['A_M'])
plt.show()

#Categorical Values
plt.hist(df['Spectral_Class'])

# ======================= HISTOGRAMS =========================


# ======================= OUTLIERS ===========================


### Outlier Detection ###
clf = LocalOutlierFactor(n_neighbors = 10)
pred = clf.fit_predict(df[['Temperature','Type']])
df[pred == -1]



sns.boxplot(x=df['Temperature'])
sns.boxplot(x=df['L'])
sns.boxplot(x=df['R'])
sns.boxplot(x=df['A_M'])


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))


df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out.shape

# ======================= OUTLIERS ===========================



# ======================= KNN ================================

#https://www.youtube.com/watch?v=LlmTZITS5vg

#Partitioning into independent variables and dependent variables (features and target feature)




#Get all columns except for last column
x = df.iloc[:, :-1].values
#Get only last column
y = df.iloc[:, -1].values



#Convert categorical values into numeric values
le = LabelEncoder()
x[:,4] = le.fit_transform(x[:,4])
x[:,5] = le.fit_transform(x[:,5])

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x = pd.DataFrame(x_scaled)



#x = x[['A_M','R']]

#Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, random_state = 0)
#What is random_state ?
#It doesn't matter if the random_state is 0 or 1 or any other integer. 
#What matters is that it should be set the same value, 
#if you want to validate your processing over multiple runs of the code.

#Normalization
#scaler = StandardScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)


#To find best k value, create a array containing 1 .... 30
k_neighbour = range(1,41)
train_acc = []
test_acc = []

#Create KNN model to find k value with highest accuracy
for k in k_neighbour:
    
    knc = KNeighborsClassifier(n_neighbors = k, weights = 'uniform', algorithm = 'auto')
    knc.fit(x_train,y_train)
    train_acc.append(knc.score(x_train,y_train))
    test_acc.append(knc.score(x_test,y_test))
    

plt.plot(k_neighbour, train_acc, label = 'Train Accuracy')
plt.plot(k_neighbour, test_acc, label = 'Test Accuracy')
plt.ylabel = 'Accuracy'
plt.xlabel('k')
plt.legend()

#Create KNN model with best K value
knc = KNeighborsClassifier(n_neighbors = (test_acc.index(max(test_acc)) + 1), weights = 'uniform', algorithm = 'auto')
#Train the model with independent and dependent variables of the train variable
knc.fit(x_train,y_train)
#Make prediction based on independent variables of the test variable
prediction = knc.predict(x_test)

#Measure the success of the model
#y_test = real variables
#prediction = predicted variables
accuracy = accuracy_score(y_test, prediction)
print('Accuracy : ',accuracy)
print(classification_report(y_test,prediction))

confusion_matrix = confusion_matrix(y_test, prediction)
print('Confusion Matrix: \n', confusion_matrix)
plot_confusion_matrix(knc, x_test, y_test, display_labels = ['0','1','2','3','4','5'])


#Cross Validation
knn = KNeighborsClassifier(n_neighbors = (test_acc.index(max(test_acc)) + 1),weights = 'uniform', algorithm = 'auto')
scores = cross_val_score (knn, x, y, cv = 10, scoring = 'accuracy')

print(scores)
print(scores.mean())

knn_df = pd.DataFrame(data = {'knn':range(10), 'accuracy' : scores} )
knn_df.plot(x = 'knn', y = 'accuracy', marker = 'o', linestyle = '--')


#Find the best K with Cross Validation
k_range = range(1,41)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores =  cross_val_score(knn,x,y, cv= 10)
    k_scores.append(scores.mean())

sns.lineplot(x = k_range, y = k_scores)

k_scores.index(max(k_scores))

#Create KNN model with best K value
knc = KNeighborsClassifier(n_neighbors = (k_scores.index(max(k_scores))
+1), weights = 'uniform', algorithm = 'auto')
#Train the model with independent and dependent variables of the train variable
knc.fit(x_train,y_train)
#Make prediction based on independent variables of the test variable
prediction = knc.predict(x_test)

#Measure the success of the model
#y_test = real variables
#prediction = predicted variables
accuracy = accuracy_score(y_test, prediction)
print('Accuracy : ',accuracy)
print(classification_report(y_test,prediction))

confusion_matrix = confusion_matrix(y_test, prediction)
print('Confusion Matrix: \n', confusion_matrix)
plot_confusion_matrix(knc, x_test, y_test, display_labels = ['0','1','2','3','4','5'])


# ======================= KNN ================================


# ======================= DECISION TREE ======================



#Format data Part 1: Split data into Dependent and Independent Variables
X = df.drop('Type', axis = 1).copy()
y = df['Type'].copy()


#Format the Data Part 2: One-Hot Encoding
#Convert categorical values into numeric values
le = LabelEncoder()
X.iloc[:,4] = le.fit_transform(X.iloc[:,4])
X.iloc[:,5] = le.fit_transform(X.iloc[:,5])

X['Spectral_Class'].value_counts()
X['Color'].value_counts()
#X_encoded = pd.get_dummies(X, columns ['Color','Spectral_Class'])


start = timeit.default_timer()
#Your statements here
stop = timeit.default_timer()
print('Time: ', stop - start)  


X = X[['A_M','R']]


#To change variables 
#y_not_zero_index = y > 0 #Get the index each non-zero value in y
#y[y_not_zero_index] = 1 #Set each non-zero value in y to 1
#y.unique() #Verify that y only contains 0 and 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10 , random_state = 0)

#Count different values of series object of pandas.core.series module
y_train.value_counts()
y_test.value_counts()

#Create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state = 0)
clf_dt = clf_dt.fit(X_train, y_train)


#NOTE: We can plot the tree and it is huge!
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt, filled=True, rounded=True, class_names = ['0','1','2','3','4','5'], 
          feature_names = X_train.columns)
#Class hangi target feature majority ise onu bastiriyor

#plot_confusion_matrix() will run test data down the tree and draw
plt.figure(figsize=(15,7.5))
plot_confusion_matrix(clf_dt, X_test, y_test, display_labels = ['0','1','2','3','4','5'])

prediction = clf_dt.predict(X_test)
print(classification_report(y_test,prediction))



scores = cross_val_score(clf_dt,X,y, cv= 10)
dt_df = pd.DataFrame(data = {'tree':range(10), 'accuracy' : scores} )
dt_df.plot(x = 'tree', y = 'accuracy', marker = 'o', linestyle = '--')






#Cost Complexity Pruning Part 1: Visualize alpha
path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
ccp_alphas = path.ccp_alphas # extract different values for alpha
ccp_alphas = ccp_alphas[:-1]# exclude the maximum value for alpha

clf_dts = []#Create an array that we will put decision trees into

#Now create one decision tree per value for alpha and store it in the array

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)
    
train_scores = [clf_dt.score (X_train,y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score (X_test,y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccp_alphas, train_scores, marker = 'o', label = 'train', drawstyle = 'steps-post')
ax.plot(ccp_alphas, test_scores, marker = 'o', label = 'test', drawstyle = 'steps-post')
ax.legend()
plt.show



#Cost Complexity Pruning Part 2: Cross Validation For Finding the Best Alpha

clf_dt = DecisionTreeClassifier(random_state= 42, ccp_alpha = 0.016)

#now use 5-fold cross validation create 5 different training and testing datasets that
#are then used to train and test the tree,
#NOTE: We use 5-fold because we don't have tons of datas

scores = cross_val_score(clf_dt,X_train,y_train, cv= 10)
dt_df = pd.DataFrame(data = {'tree':range(10), 'accuracy' : scores} )
dt_df.plot(x = 'tree', y = 'accuracy', marker = 'o', linestyle = '--')



# ======================= DECISION TREE ======================

 