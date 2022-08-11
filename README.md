# Table of Contents

* [Introduction](#Introduction)
* [Data Visualization](#Data_Visualization)
* [Data Preparation](#Data_Preparation) 
   * [Missing Values](#Missing_Values)
   * [Distribution of Values](#Distribution_of_Values)
   * [Duplicated Rows](#Duplicated_Rows)
* [Models](#Models) 
   * [Decision Tree Model](#Decision_Tree_Model)
   * [KNN Model](#KNN_Model)
* [CONCLUSION](#CONCLUSION) 




# Stellar-Classification
A machine learning project to classify stars.

# Introduction <a class="anchor" id="Introduction"></a>

Stellar classification is the classification of stars based on their characteristic features. Stellar classification is one of the subtopics of Astronomy. The dataset contains 7 features (columns) for each star. These features are:

1)	Temperature (Numeric Values in Kelvin)
2)	Relative Luminosity (Numeric Values)
3)	Relative Radius (Numeric Values)
4)	Absolute Magnitude (Numeric Values)
5)	General Observed Color (Nominal)
6)	Spectral Class (Nominal)
7)	Type (Target Feature – Discrete Numeric)

Possible values of Type target feature are 0, 1, 2, 3, 4, 5.

Red dwarf – 0
Brown dwarf – 1
White dwarf – 2
Main Sequence – 3
Super Giant – 4
Hyper Giant – 5

# Data Visualization <a class="anchor" id="Data_Visualization"></a>

Data visualization plays a critical role in understanding the dataset we work on. It may give us hints.

One way of data visualization is to draw scatter plots. Scatter plots shows us where each data stand in 2D coordinate system based on 2 attributes(columns). The figure below shows instances based on ‘A_M’ column and ‘Spectral Class’ column.

![image](https://user-images.githubusercontent.com/50465232/184242870-2b2418dd-3edd-49f0-8a46-5a85fdc8503d.png)

We see that instances split based on their values. It might be frustrating to draw scatter plot for each column pairs if dataset contains so many columns. Luckily, there are some functions that do this for each pair automatically. For this example, seaborn python library has been used.

![image](https://user-images.githubusercontent.com/50465232/184243303-fc95fdab-ff02-465b-8b7b-7339b06917d2.png)

The matrix below shows the scatter plots and value chart graphs. Intersection points show scatter plots if the other column is different than row. If the row and column names are equal than it shows value chart graph.
Examining these plots, we can select only some features to train our data to save time and resources. This is called feature selection.


# Data Preparation <a class="anchor" id="Data_Preparation"></a>

As the first step, the dataset has been imported from a .csv file by using pandas library. Total number of rows and columns have been found out by using shape method. There are 240 rows and 7 attributes.

2.1 Missing Values <a class="anchor" id="Missing_Values"></a>

Whether the dataset contains any missing values has been investigated. For this operation, “info()” function from pandas library has been used. The dataset contains no missing values. This function also shows data types for each feature. Object is usually string.

![image](https://user-images.githubusercontent.com/50465232/184244182-0dc788c8-35b6-4fc3-ae58-caef2e8127df.png)


2.2 Distribution of Values <a class="anchor" id="Distribution_of_Values"></a>

Distribution of values for each feature has been investigated to observe outliers or faults.

![image](https://user-images.githubusercontent.com/50465232/184244291-c827a9f9-1683-4d96-82f4-678a54f6f63a.png)

Type column is the target feature column. There are 40 instances for each unique value. 

![image](https://user-images.githubusercontent.com/50465232/184244374-f08d1eea-8a15-4600-ae7b-910109b55568.png)

There is no mistake in spectral class column.

![image](https://user-images.githubusercontent.com/50465232/184244410-c3b3ef6c-869d-4f21-8349-f6c84130d6dd.png)

There are some mistakes in color column values for categorical features. For example, there are 4 different names for ‘blue-white’.
1)	‘Blue-white’
2)	‘Blue White’
3)	‘Blue white’
4)	‘Blue-White’
After correcting each one:

![image](https://user-images.githubusercontent.com/50465232/184244456-bc6e934b-3d5a-4a2a-b7a3-9905b5cf5417.png)

Now draw boxplot for each continious numeric feature.

![image](https://user-images.githubusercontent.com/50465232/184244497-4d57e152-204f-484b-bea6-c13bc0618fb4.png)

![image](https://user-images.githubusercontent.com/50465232/184244508-c17738bc-230c-4897-9538-1e813c929082.png)

![image](https://user-images.githubusercontent.com/50465232/184244529-85996984-01f3-437f-9053-7189627fd37e.png)

![image](https://user-images.githubusercontent.com/50465232/184244538-cba35848-bce4-47ef-8c2b-e49594435352.png)

There are many outliers. Outliers should be removed or corrected.

2.3 Duplicated Rows <a class="anchor" id="Duplicated_Rows"></a>

To check for duplicated rows, duplicated() function has been used. This function returned ‘false’ for each row, meaning there are no duplicated rows for this dataset.


# Models <a class="anchor" id="Models"></a>

## 3.1 Decision Tree Model <a class="anchor" id="Decision_Tree_Model"></a>

Decision tree is based on partitioning the dataset by descriptive feature that has highest information gain. 

The first step of building decision tree model is the partition of dataset as dependent and independent variables.
After this, categorical variables such as color and spectral class values has been converted into numeric values. This operation is called label encoding. The figure below shows how variables have changed.

![image](https://user-images.githubusercontent.com/50465232/184245188-1e59453a-73a3-452c-8880-f6cd6af200b0.png)

![image](https://user-images.githubusercontent.com/50465232/184245202-7e729635-d6b2-4b80-a02f-92ee3917bc9f.png)

Machine learning models require test and training dataset. Split the dataset into test and training. The ratio size of test set is 0.10. The ratio size of training set is 0.90.
Figures below show how many instances from each unique target feature value have been taken.

![image](https://user-images.githubusercontent.com/50465232/184245261-ba72bf82-30d3-4771-8a85-fff5b3c839c1.png)

![image](https://user-images.githubusercontent.com/50465232/184245274-e4b56174-8dc4-448b-8e86-5e3c5bc3c805.png)

Now all is ready to create decision tree model. After creating it and fitting with train data, the model can be plotted.

![image](https://user-images.githubusercontent.com/50465232/184245330-a1878ffd-f6e1-4762-a7a4-7cf3fe82985a.png)

The first line states by which column the model split the dataset if it splits.
Decision tree is partitioned by using Gini index. This comes default with the model.
Third row shows how the values for type column are distributed.
Last row shows which type value is majority in each node.
This figure shows only 2 features has been used to build decision tree model.

Testing results are given in the confusion matrix figure below.
![image](https://user-images.githubusercontent.com/50465232/184245388-754599e0-9a1d-4cd2-bd8e-3b35ff39e1eb.png)

From this confusion matrix we see that accuracy of the model is 1.0.
Other measure variables are shown below.

![image](https://user-images.githubusercontent.com/50465232/184245416-b4045510-f1aa-4c1f-8728-d6e18bca5e1f.png)

To make sure our model does not overfitting or underfitting, cross validation is used.

![image](https://user-images.githubusercontent.com/50465232/184245470-0606847c-8b95-48af-8726-7832a3582e39.png)

For all 10 folds, the result is 1.0 accuracy.

Building and testing the model take approximately 0.4 seconds.
By examining the tree graph and scatter plots in data visualization part, we can do feature selection.
In tree graph, only R and A_M features had been used. So before splitting dataset into train and test, take only these features as independent variables. After same operations has been done, the runtime drops to 0.3 seconds while accuracy value does not change.

If there was more column, the runtime would drop more.


# 3.2 KNN Model <a class="anchor" id="KNN_Model"></a>

K-Nearest Neighbor is a non-parametric machine learning method which is used in both classification and regression. As its name states, prediction is made by summing distances between existing instances and new instances. K number of ordered values (ascending) are checked by their class values. Highest frequency class value is selected as prediction of the new instance.  

Data preparation for KNN is exactly same as decision tree model. Splitting data into independent variables and dependent variables and then split as train and test sets. Before creating our model, the best value for k should be found. For this, different models are created for a set of k values and their performances are examined.

![image](https://user-images.githubusercontent.com/50465232/184245594-abe3af21-ee65-47a6-b631-4590f9a1bcc5.png)

The figure above shows accuracy values for each k value ranking from 1 to 40 on test and train dataset. After trying the highest value in test accuracy, we get the following results.

![image](https://user-images.githubusercontent.com/50465232/184245640-acba4906-8f75-4bcb-804f-5ecd62fb95e3.png)


![image](https://user-images.githubusercontent.com/50465232/184245666-2833c080-2058-4ec8-8101-4f3f0e05e439.png)


The results are not good because some values in the dataset way higher than the others. 

For example:

![image](https://user-images.githubusercontent.com/50465232/184245716-be41e3de-d01d-432a-ad2a-1fa4a42fedf7.png)

Value distribution for columns is not equally distributed. To overcome this, normalization operation should be done before creating the mode. After normalization operation, accuracy values for train and test datasets becomes:

![image](https://user-images.githubusercontent.com/50465232/184245783-70f01a42-c9ec-4cbf-b2da-c07fedcb4bb2.png)


After model creation with highest k value in the test dataset and testing operation:

![image](https://user-images.githubusercontent.com/50465232/184245825-46882d89-337f-46c9-b2c4-a8be77c153ac.png)

![image](https://user-images.githubusercontent.com/50465232/184245847-dd83870c-6972-4dba-9551-e954f29179cc.png)

To make sure the model is successful, or it was just luck, cross validation operation has been applied. 

The plot below shows accuracy values for each fold out of 10.

![image](https://user-images.githubusercontent.com/50465232/184245905-204ad900-fac8-4759-bcb9-f4de33385c55.png)

The mean is 0.95 so the model is successful.

To find the best k value with cross validation, do cross validation operation for each k value ranged from 1 to 40. The figure below shows how the accuracy changes.

![image](https://user-images.githubusercontent.com/50465232/184245972-49a1aca0-d78c-42d7-99d3-0bad2cd5995a.png)

Now select this k value to predict test dataset.

![image](https://user-images.githubusercontent.com/50465232/184246021-f9d7cfd8-6b20-472a-9bba-4621e5674c9a.png)

![image](https://user-images.githubusercontent.com/50465232/184246054-2de595a3-7811-4658-9d37-69a83eed0bc6.png)

The result is almost same. But this result is more trustworthy than the first result where accuracy value is 1.

# CONCLUSION <a class="anchor" id="CONCLUSION"></a>

The models almost equal accuracy rates so it better to look at their runtime. Decision tree runs faster than KNN model and cross validation score of decision tree for each fold is 1 as KNN model’s is not.
So we can say that decision tree is more suitable than KNN for this dataset.















