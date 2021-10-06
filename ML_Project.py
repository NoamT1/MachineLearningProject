
#importing libaries used in this project
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

train = pd.read_csv("train.csv")  #import the training dataset

# Part 1 - Data Exploration
#Exploring nulls and duplicates

print(train.isnull().sum())
print('\nThe total number of missing values:',sum(train.isnull().sum()))
dat = train.isnull().sum()
df_labels=train['label'] 

#null values in both 14 and 19 column
train.reset_index()
print('\nFor the features that have relatively many null values, we checked whether there is an overlap in places where the values\nare empty so that if the overlap is large we may decide to delete the empty value rows,\nas these are not many lines relative to the total number of rows in the data set.')
features = ['14','19']
print('\nFor each pair the ovarlap of missing values:')
pairs = list(itertools.combinations(features, 2))
for pair in pairs:
    check= train[[pair[0],pair[1]]]
    print(pair,"-> number of mutual null rows: ",check[(check[pair[0]].isnull())&(check[pair[1]].isnull())].shape[0])

# lines that are completely identical  
num_of_duplicates = sum(train.duplicated())
print (f"{num_of_duplicates} duplicates found") if num_of_duplicates > 0 else print('no duplicates')

# Exploring features
#we want to know what % of the labels is tagged as 1 
# num of rows minus sum of values in the column divided by num of rows
a=(train.label.count()-train.label.sum())/(train.label.count())*100 
#percentage calculation of zeros - how many 0s we have out of the entire data 
print(f"{round(a)} % are tagged as 0")
train.label.value_counts().plot.barh()

#Visualization:
train_without_nulls = train.dropna() #Remove missing values for the visualization
train_features = train_without_nulls.loc[:,:'20']  #Access a group of rows and columns by label(s) or a boolean array. features array
train_labels = pd.DataFrame(train['label'])  # label column 
numeric_columns=list(train_features.columns[train_features.dtypes != 'object']) #separate numeric and categorial farture
categorial_columns=list(train_features.columns[train_features.dtypes == 'object']) 

#For the numerical features - plot histogram and box plot
def numerical_features_visualization(numeric_columns):
     for column in numeric_columns:
         plt.figure(figsize = (25,5))
         plt.subplot(131)
         train_features[column].hist()
         plt.title('Histogram plot- feature number '+column)
         plt.xlabel('value')
         plt.subplot(132)
         train_without_nulls[[column]].boxplot()
         plt.title('Box plot- feature number '+column)
         plt.show()
        
#For the categorial features - plot that showing the amount of each category
def categorial_features_visualization(categorial_columns):
     for column in categorial_columns:
         fig, ax =plt.subplots(figsize=(18,5))
         sns.countplot(train_without_nulls[column])
         plt.title('Plot- Feature number'+column)
         plt.xlabel('Categories')
         plt.show()
         
numerical_features_visualization(numeric_columns)
categorial_features_visualization(categorial_columns)

#Heat Map
pairs_heatmap_cor = train_features.corr() 
plt.subplots(figsize=(18, 18))
ax = sns.heatmap(pairs_heatmap_cor, annot = True,cmap= 'coolwarm')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.7, top - 0.7)
ax.set_title('Heat Map: Correlation between the features')

#Pie charts:

#pie charts: feature 1:
y_val = train['1'].value_counts().index.tolist()
counts = train['1'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()

#pie charts: feature 2:
y_val = train['2'].value_counts().index.tolist()
counts = train['2'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()
"""
we have decided that we keep feature#2 as a numerical descrete feature, 
since there are so many different values that it may show. 
"""
#we can see that -1d is the most common value. In addition, there are many different categories. 

#pie charts: feature 6:
y_val = train['6'].value_counts().index.tolist()
counts = train['6'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()

#pie charts: feature 12:
y_val = train['12'].value_counts().index.tolist()
counts = train['12'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()

#pie charts: feature 16:
y_val = train['16'].value_counts().index.tolist()
counts = train['16'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()

#pie charts: feature 18:
y_val = train['18'].value_counts().index.tolist()
counts = train['18'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()

#pie charts: feature 19:
y_val = train['19'].value_counts().index.tolist()
counts = train['19'].value_counts().values.tolist()
fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=y_val, autopct='%1.1f%%')
ax1.axis([0, 2, -1, 0.4])
plt.show()

#Creating a correlation matrix between those features:
#Those 5 features are all categorial
df=pd.DataFrame({'1':train['1'],'2':train['2'],'12':train['12'],'16': train['16'],'18':train['18']})
df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

# Part 2- Pre-processing
#We will now apply our pre-processing methods on both training and test datasets. Importing the test dataset:

test = pd.read_csv("test_without_target.csv") 
test=test.loc[:,"0":] #dropping the first column because it contains indexes which do not overlap with the train set data 
test.head()

#removing columns as explained above:
train=train.drop(columns=['9','17','19'])
test=test.drop(columns=['9','17','19'])

#Pre-processing, stage 1 - Removing *outliers*

#removing outliers 
def remove_numeric_outliers(x):
    treshold=x.shape[0]*0.01    #all the observations
    for col in list(x.columns[x.dtypes != 'object']):   # only numeric columns
        #number of rows that are outside the 3 standard deviations :
        num_of_rows = x[(np.abs(stats.zscore(x[[col]]))>3).all(axis=1)].shape[0] 
        if  0 < num_of_rows < treshold :
            print ('Number of outliers remove from column ', col, ' is: %d'% num_of_rows)
            x= x[(np.abs(stats.zscore(x[[col]]))<3).all(axis=1)]  #removing the outlier
    return x
    
train=remove_numeric_outliers(train)

#Preprocessing, stage 2 - Feature Display
#Some features required modifications to their display in order to be used correctly. More specifically, the function below converts feature 12 from n/y to 0/1, and removes the letter d from the end of feature 2 and the letter a from feature 18.

#Editing values so they can match to the others in their column
def feature_display(df):
    df['12'] = df['12'].replace(['n'],'0').replace(['y'],'1')
    df['2'] = df['2'].str.strip('d') #need to further discuss 
   #  df['18'] = df['18'].str.strip('a')
   #decided not to do this for the 18th column
    return df

train = feature_display(train)
train.head()

test=feature_display(test)
test.head()

numeric_features=['3','4',"5","7","8","10"]
categorial_variables=['1','2','6','12','16']

#Preprocessing, stage 3 - Normalization
#Since no max or min of the data (such as happens with image processing), we choose to scale the data using the *Z-score standardisation* method.
#NORMALIZATION

#organize the indexes below or delete, all numerical
categorial_indexes = ['1','6','16','18'] #those are the dummies-affected columns or "binary-categorial" cols that dont need normalization
numeric_indexes = ['0','2','3','4','5','7','8','10','11','12','13','14','15','20']

def normalization(train,test,numeric_indexes):   
    #create datasets with only numerical columns 
    train_numeric = train[numeric_indexes]
    test_numeric = test[numeric_indexes]
    # normalize numeric columns in both datasets
    standard_scaler = StandardScaler()              #we initialize our scaler
    standard_scaler.fit(train_numeric)              #fit according to our scaler (which is the train data)
    normalized_train_numeric = standard_scaler.transform(train_numeric.values)   #transform train df
    normalized_test_numeric = standard_scaler.transform(train_numeric.values)    #tranform test df
    # convert train and test from numpy objects to pandas
    normalized_train_numeric = pd.DataFrame(data=normalized_train_numeric[1:,:],index=None,columns=numeric_indexes) 
    normalized_test_numeric = pd.DataFrame(data=normalized_test_numeric[1:,:],index=None,columns=numeric_indexes) 
    # Replace the numeric columns of the original datasets with the normalized value
    for i in numeric_indexes:
        train[i] = normalized_train_numeric[i]
        test[i] = normalized_test_numeric[i]
    return train,test

train=train.fillna(value=0) 
train_numeric=train.loc[:, numeric_indexes]
test_numeric=test.loc[:, numeric_indexes]
train_numeric,test_numeric = normalization(train_numeric,test_numeric,numeric_indexes)

train_categorial=train[categorial_indexes]
print (train_categorial.isnull().sum().sum())
test_categorial=test[categorial_indexes]

#Preprocessing, stage 4 - Filling missing values

#filling missing values 
def fill_median_value(df):
    global numeric_indexes
    #fillig missing numeric values with the median :
    #numeric_columns = list(df.columns[df.dtypes != 'object'])
    for column in numeric_indexes:
        # Transfer column to independent series
        col_data = df[column]
        # Look to see if there is any missing numerical data
        missing_data = sum(col_data.isna())
        if missing_data > 0:
            # Get median and replace missing numerical data with median
            col_median = col_data.median()
            col_data.fillna(col_median, inplace=True)
            df[column] = col_data
    return df

def fill_most_common_value(df):
    global categorial_indexes
    #fillig missing categorial values with the most common value:
    for column in categorial_indexes:
        col_data = df[column]
        # Look to see if there is any missing data
        missing_data = sum(col_data.isnull())
        if missing_data > 0:
            # Get median and replace missing data with the most common category
            col_max = (train.groupby(column).size()).idxmax()
            col_data.fillna(col_max, inplace=True)
            df[column] = col_data
    return df 

train.isnull().sum().sum()
train_numeric=fill_median_value(train_numeric)
train_categorial=fill_most_common_value(train_categorial)
test_numeric=fill_median_value(test_numeric)
test_categorial=fill_most_common_value(test_categorial)

# splitting with get_dummies:

dummies_lst=['1','6','16','18'] #the columns in the dataset we want to split using the get_dumies function

train_categorial = pd.get_dummies(train.loc[:, dummies_lst])
test_categorial = pd.get_dummies(test.loc[:, dummies_lst])

#Preprocessing, stage 5 - Reducting dimensions
#5 - a. Manual removal
#We start off by manually removing features to reduct the dimensionality of the dataset. The rationale and the execution of this removal can be found at the beginning of the preprocessing part.

train_categorial=train_categorial.drop(columns=['1_0','6_0','16_0','18_0'])

# joining the categorial and numeric columns back into complete datasets
train_pca = pd.concat([train_categorial, train_numeric], axis=1)
test_pca=pd.concat([test_categorial, test_numeric], axis=1)

#We will now create 3 different options of the train dataset, each with different dimensions. Then we will compare between their perfromance measurements.
#5 - b. PCA

def pca_func(train_set,test_set,tot_var):
    pca = PCA(tot_var)
    pca.fit(train_set)
    transformed_train_set = pd.DataFrame(pca.transform(train_set))
    transformed_test_set = pd.DataFrame(pca.transform(test_set))
    print ('Before PCA: There were '+ str(pca.n_features_) +' features in the dataframe,'
           ' and after the PCA procedure we have ' +str(transformed_train_set.shape[1])+' features.')
    return transformed_train_set,transformed_test_set, 
    
train_after_pca,test_after_pca = pca_func(train_pca,test_pca,0.95)
#As we see, there is not a variable which explains a significant portion of the variance.

#Part 3 - Model Setup
#In this section we will define several models, in order to test which of those is best for our problem.

#define a gridsearch function for us to use down the road
def GridSearch(model,initialParams):
    global train_after_pca
    #kfold = sklearn.model_selection.KFold()  
    kfold = sklearn.model_selection.KFold(n_splits = 10, random_state = 42, shuffle=True)  #what is random state??
    GridSearch_instance = GridSearchCV(model, initialParams, cv = kfold, scoring='roc_auc') #initiate the gridsearch. We use the default Kfold values (K=5, shuffle off), and GridSearch's scoring is determined by AUC measurement
    GridSearch_instance.fit(train_after_pca,df_labels) #apply the gridsearch on our data
    
    #return the best parameters found by Gridsearch
    return GridSearch_instance.best_params_

#Logistic Regression
# First, we define initial parmeters for this model:
Cs = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]
initialParams = {'C':Cs, # Inverse regularization parameter, larger C means less regularization.
                  'penalty':['l1', 'l2'],  # l1 stands for abs distance, l2 stands for quad_distances
                  'solver':['liblinear'], # Algorithm to use in the optimization problem
                  'tol':[0.0001], # Stopping critiria
                  'max_iter':[100]} 
                
# Then we gridsearch in order to find the best parameteres for our dataset
LogisticReg_BestParams=GridSearch(LogisticRegression(),initialParams)


#Naive Bayes Classifier
# First, we define initial parmeters for this model:
initialParams = {'priors':[None],'var_smoothing':[1e-1,1e-2,1e-3,1e-4]}  

# Then we gridsearch in order to find the best parameteres for our dataset
NaiveBayes_BestParams = GridSearch(GaussianNB(),initialParams)

#Adaptive Boosting

#n_samples should be len(train_after_pca), switch it if you want a faster runtime
train_after_pca, df_labels = make_classification(n_samples=len(train_after_pca), 
                                                 n_features=22,n_informative=2, 
                                                  n_redundant=0,random_state=42, shuffle=True)


#set initial parameters for the adaptive booster
initialParams = {'algorithm':['SAMME'], 
                  'learning_rate':[0.1,0.01],
                  'n_estimators': [30,40,50,60,100],
                  'random_state':[0]}

#perform Gridsearch using the initial parameters, in order to find the best possible we can                                              
AdaptiveBoost_BestParams = GridSearch(AdaBoostClassifier(), initialParams)


#Decision Tree 

initialParams = {'criterion':["gini", "entropy"],
                    'max_depth':[None],      #default, the maximum depth of the tree (None: Unlimited). The higher it gets, MORE fit
                    'min_samples_split':[2], #default, what is the minimum samples required to keep splitting. The higher it gets, LESS fit
                    'max_features':[None],   #default, how many features are we allowed to use (None: Unlimited). The higher it gets, MORE Fit
                    'max_leaf_nodes':[None], #default, the maximum number of leaves (None: Unlimited). The higher it gets, MORE fit.
                    #'min_impurity_decrease':[0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9], #A node will be split if this split induces a decrease of the impurity greater than or equal to this value
                    #'min_samples_leaf':[1],           #default
                    #'min_weight_fraction_leaf':[0.0], #default
                    #'random_state':[42],              #seed
                    'min_impurity_split':[1e-7],      #default
                    #'class_weight':[None],            #default
                    #'presort':[False]                #default
                       }

DecisionTreeClassifier_BestParams = GridSearch(DecisionTreeClassifier(), initialParams)
# Part 4 - Model Evaluation
#In this section we will define several models, in order to test which of those is best for our problem.
# The classifiers we'll be evaluating in this part
classifiers = [GaussianNB(**NaiveBayes_BestParams), 
               LogisticRegression(**LogisticReg_BestParams), 
               AdaBoostClassifier(**AdaptiveBoost_BestParams),
               DecisionTreeClassifier(**DecisionTreeClassifier_BestParams)]


clf_names = ["Gaussian Naive Bayes", "Logistic Regression", "Adaptive Booster", "Decision Tree"]

print("The classifiers are:")
print(clf_names)
print("Their parameters are:")
print(classifiers)

#KFoldPlot

def KfoldPlot(train_df, df_labels, classifiers, clf_names, k=10):  
    # setting K to be 10 folds by default, but still proviting the option to experiement with K if we'd like
    # Preparing some variables and a plot figure to be used in this function
    meanAucArrayTrain, meanAucArrayValidation = [] , []
    plt.figure(figsize = (10,10))
    counter = 0    
    for classifier in classifiers:        
        # Resetting mean tpr (true-positives) and fpr (false-positives), to be used later in this loop
        meanTprTrain = 0.0
        meanFprTrain = np.linspace(0, 1, 100)
        meanTprValidation = 0.0
        meanFprValidation = np.linspace(0, 1, 100)

        # Initializing KFold
        KF = KFold(n_splits=k, shuffle=True) 
        # Splitting our data by applying the KFold , and iterating on each fold
        for train_index , validation_index in KF.split(train_df):
            # Using the fold's indices to split the data and its outcome labels into a train dataset and a validation dataset
            train_traindata, train_validation = train_df[train_index], train_df[validation_index]
            labels_traindata, labels_validation = df_labels[train_index], df_labels[validation_index]
            train_traindata = pd.DataFrame(train_traindata)
            labels_traindata=pd.DataFrame(labels_traindata)
            labels_validation=pd.DataFrame(labels_validation)

            #apply the current classifier on these datasets
            classifier.fit(train_traindata, labels_traindata)

            #Predicting using this classifier: both for test and validation datasets
            prob_prediction_train = classifier.predict_proba(train_traindata)[:, 1]
            
            # Creating the ROC curve for train dataset (for this fold), and adding it to the array of means of true positives
            fpr_train, tpr_train, thresholds_train = sklearn.metrics.roc_curve(labels_traindata, prob_prediction_train, drop_intermediate=True)
            meanTprTrain += np.interp(meanFprTrain, fpr_train, tpr_train)
            meanTprTrain[0] = 0.0 # (resetting the first object of the array)

            # Same for the validation dataset
            prob_prediction_validation = classifier.predict_proba(train_validation)[:, 1]
            fpr_validation, tpr_validation, thresholds_validation = sklearn.metrics.roc_curve(labels_validation, prob_prediction_validation, drop_intermediate=True)
            meanTprValidation += np.interp(meanFprValidation, fpr_validation, tpr_validation)
            meanTprValidation[0] = 0.0
            
         # Preparing the arrays for mean TPR and AUC  
        meanTprTrain = meanTprTrain / k  # dividing mean_tpr by number of folds to get actual means
        meanTprValidation = meanTprValidation / k 
        meanAucTrain = sklearn.metrics.auc(meanFprTrain, meanTprTrain)
        meanAucValidation = sklearn.metrics.auc(meanFprValidation, meanTprValidation)
        meanAucArrayTrain.append(meanAucTrain) 
        meanAucArrayValidation.append(meanAucValidation)        
        
        # Plotting the mean roc
        current_clf_name = clf_names[counter]
        counter += 1
        plt.plot(meanFprValidation, meanTprValidation, linestyle='-', label='Mean ROC for %s (area = %0.3f)' % (current_clf_name,meanAucValidation))

        #Checking for overfitting for the current classifier
        delta = np.abs(meanAucTrain - meanAucValidation)
        if delta > 0.1: # This is the value we determined for overfitting
            print('there might be an overfitting with the model',clf_names[counter])

     #After passing all the models, we will present one ROC graph including the average auc of each model
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('FPR - False Positive Rate')
    plt.ylabel('TPR - True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    print (meanAucArrayValidation, meanAucArrayTrain)
    return meanAucArrayValidation, meanAucArrayTrain
    
#37
 # Running the KFoldPlot

meanAucArrayValidation, meanAucArrayTrain = KfoldPlot(train_after_pca, df_labels, classifiers, clf_names)

#Best model: 

#Choose the best Classifier model according to the AUC value
max_auc_index = meanAucArrayValidation.index(np.max(meanAucArrayValidation))
best_clf = classifiers[max_auc_index]
best_clfname = clf_names[max_auc_index]
print('The best classifier is', best_clfname,'with an average AUC score of', np.max(meanAucArrayValidation))

#Confusion matrix 
# the function "train_test_split" splits the each dataset into two parts that are 
# intended for training and validation in the confusion matrix

X_for_train, X_for_validation, y_train, y_valid = sklearn.model_selection.train_test_split(train_after_pca, df_labels, 
                                                                                           test_size=0.3, shuffle=True,
                                                                                           random_state=42)
acc_list=[]
counter = 0

# looping on each model in order to create a confusion matrix
for model in classifiers:
    #fitting the model
    model.fit(X_for_train,y_train)

    #finding model's name
    clfname = clf_names[counter]
    counter +=1

    # predicting
    y_predict=model.predict(X_for_validation)

    #creating a confusion matrix instance, and setting the true negatives, false positive, false negatives, true positives
    cm = confusion_matrix(y_valid,y_predict)
    tn, fp, fn, tp=cm.ravel()
   
    # plotting the confusion matrix itself
    confusion_matrix_plot = np.array([[fn,tn],[tp,fp]])
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(confusion_matrix_plot, annot=True, fmt='d',annot_kws={'size':16},cmap="BuPu")
    ax.set_ylim([0,2])
    plt.ylabel('Predicted values')
    plt.xlabel('Actual values')
    plt.title('Confusion matrix for model %s' % clfname)
    ax.xaxis.set_ticklabels(['1', '0']); ax.yaxis.set_ticklabels(['0', '1'])
    plt.show()

# Part 5 - Prediction

# a variable to store the results
result = pd.DataFrame()          
            
# fitting the best model to the train data
best_clf.fit(train_after_pca,df_labels)

# applying this classifier on the test data
result['pred_proba'] = best_clf.predict_proba(test_after_pca)[:,1]

# submission!
result.to_csv('Submission_group_26.csv')
