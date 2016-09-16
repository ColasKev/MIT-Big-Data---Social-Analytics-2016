
# coding: utf-8

# <div align="right">Python 2.7 Jupyter Notebook</div>
# 
# # Predictive Modeling Using Bandicoot Features
# <br>
# <div class="alert alert-warning">
# <b>This notebook contains exercises for students completing the non-technical track.</b> Note that you will not be disadvantaged for completing only the non-technical notebooks. If you are a more technical student and wish to deepen your understanding and qualify for additional credit, we recommend that you complete the technical notebooks instead.
# </div>
# 
# ### Your completion of the notebook exercises will be graded based on your ability to: 
# 
# > **Understand**: Does your pseudo-code and/or comments show evidence that you recall and understand technical concepts?
# 
# > **Apply**: Are you able to execute code, using the supplied examples, that perform the required functionality on supplied or generated data sets? 
# 
# > **Analyze**: Are you able to pick the relevant method, library or resolve specific stated questions?
# 
# > **Evaluate**: Are you able to interpret the results and justify your interpretation based on the observed data?
# 
# > **Create**: Your ability to produce notebooks that serve as computational record of a session that can be used to share your insights with others? 

# # Notebook introduction
# This notebook serve as a brief introduction to machine learning. It serves as an introduction to this exciting field and provides you with hands on examples of the various steps you are likely to perform. We utilize a number of the Python libraries and methods introduced in earlier modules to generate features and compute behavioral indicators.
# 
# > **Note**:
# 
# > Many of to terms introduced will be new to many of the students, but will make more sense when you reach the end of the notebook.
# 
# The approach followed in this notebook entails computing a number of behavioral indicators, defining the target variable before going back to steps introduced in previous modules such as performing exploratory data analysis and preprocessing of data. Classification with cross validation is followed by a section on the the receiver operation characteristic (ROC) curve. In the final section we touch on optimal classifier parameters.

# # 1. Introduction to Machine Learning in Python
# In the previous module we evaluated more 1400 behavioral indicators for many users in the Friends and Family dataset. The next logical questions to ask is what are these indicators good for?  How can you process or make sense of so many variables or features? The short answer is without the help of computers it is virtually impossible. Thanks to **machine learning** we can extract answers to meaningful questions using such vast amounts of data. Examples of such questions that can be of interest include  the following; 
# - What is a person's gender?  
# - How susceptible to marketing are they? 
# - Are the patterns in data suggesting a higher probability that the person will not be making use of our service? 
# - What is the propensity of them taking up a new product or service offering? 
# - Who is best positioned to assist them should they engage with our organization through any one of our channels? 
# - How does their usage patterns inform product design and development?
# - Who in a community are most at risk to an epidemic or disease outbreak?
# 
# 
# As humans amass huge volumes of data, machine learning is increasingly become central in being able to infer and make prediction from this data. 
# 
# ### What is Machine Learning?
# 
# According to the [SAS institute](http://www.sas.com/en_us/insights/analytics/machine-learning.html), *machine learning is a method of data analysis that automates analytical model building. Using algorithms that iteratively learn from data, machine learning allows computers to find hidden insights without being explicitly programmed where to look*. There are two main classes of machine learning algorithms: (i) supervised and (ii) unsupervised learning. 
# 
# In unsupervised learning, the objective is to identify or infer a function to describe hidden structure or similarity of patterns in unlabeled data. In module 4, we were introduced to three different methods that can be used for clustering and community detection of graph networks, namely hierarchical clustering, modularity maximization, and spectral graph partitioning. These methods belong to the unsupervised learning family of machine learning algorithms.
# 
# In supervised learning, which is the focus of this notebook, not only is one provided with a set of features ( $ {X}_i$, for $i=1,...,N$) but also a set of labels ${y}_i, i=1,...,N$, where each $y_i$ corresponds to $X_i$. To give a concrete example, we have already been introduced to the very rich set of features one can derive from mobile phone data. Basic demographic or economic information is typically missing from these data sets, such as economic status of an individual, or their gender or age. If such information is available for a subset of the population in our database, an interesting challenge with broad policy implications is to use mobile phone data in predicting these demographic/economic/social indicators where such information is unavailable or not easily accessible. 
# 
# Thus, in supervised learning, one uses a given pair of input data and a corresponding supervisory target or desired target, $(y,X)$ to learn a function $f$ that can be used to predict the unknown target value of some input vector, i.e.
# $$ y = f(X).$$
# 
# The learning process also uses some specified objective function to optimize when fitting to data. A simple is example of an objective function is the sum of squares error formula used in linear regression that you may be familiar with. The error term itself is usually referred to as the *[loss function](http://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing)* in machine learning, and performance of a given algorithm is commonly reported in terms of the loss function employed. Some of these will be discussed later in the notebook.
# 
# Exactly what does learning entail? At its most basic, learning involves specifying a model structure $f$ that hopefully can extract regularities for the data or problem at hand as well as the appropriate objective function to optimize using a specified loss function. Learning (or fitting) the model essentially means finding optimal parameters of the model structure using provided input/target data. This is also called *training* the model. It is common (and best practice) to split the provided data into at least two sets - training and test data sets. The rationale for this split is the following. We are fitting the model to perform as well as it can using given or seen data. However, our goal is for the model to perform well on *unseen* data, that is data we still have to collect but whose labels ($y$) we do not know. To manage this challenge, we keep part of data hidden from the training or fitting process. Once satisfied we have an optimal model, we can then assess how well the model *generalizes* on unseen data using the data that was hidden from the training process. Data used to fit the model is referred to as training data whilst data kept hidden from the training but used to assess generalization performance of the model is called test data. There are different variations of this train/test split. You may also encounter the term validation data being used for essentially the same objective. (There are some technicalities that go beyond the scope of this course.)  
# 
# 
# Typically, one collects data of known $(y, X)$ pairs that hopefully captures the range of variability and associated outcomes for the domain problem on hand. Learning a function $f$ essentially involves capturing statistical regularities from systems that are potentially very complex and highly dimensional. Unlike classical scientific approaches, where the objective is finding a comprehensible law or theory to describe the nature or mechanics, in machine learning, the learned function $f$ is typically a complicated mathematical expression that describes the data e.g. the behavior of a economically disadvantaged person as observed through his telephone activity in a specific country.
# 
# Python has a growing set of well developed toolkits that make it easier to perform this kind of data analysis, such as the [Scikit-Learn](http://scikit-learn.org/stable/) package commonly refered to as sklearn and [Mlxtend](http://rasbt.github.io/mlxtend/) (machine learning extensions), amongst others.

# <br>
# <div class="alert alert-info">
# <b>Exercise 1 Start.</b>
# </div>
# 
# 
# ### Instructions
# Please provide brief written answers in markdown to demonstrate your understanding of the four concepts below:
#  
#  > 1. What is machine learning?
#  
#  > 2. How can machine learning be used?
#  
#  > 3. What is overfitting?
#  
#  > 4. How can you prevent overfitting?
# 
# 
# > **Hint**: Please review [this article](http://www.ma.utexas.edu/users/mks/statmistakes/ovefitting.html) as well as the [Wikipedia](https://en.wikipedia.org/wiki/Overfitting) page for additional information on overfitting.
# 

# A. The definition is above but it is a method of data analysis that automates methodical and documented analytical model building and analysis. Instead of using data collected by surveys or interviews, this data analysis method allows researchers and corporate scientists to iteratively learn from data collected, and find without human biases hidden insights without being explicitly programmed where to look. There are two main classes of machine learning algorithms: unsupervised, semi-supervised, supervised and reinforcement learning. 
# 
# In unsupervised learning, the objective is to identify or infer a function to describe hidden structure or similarity of patterns in unlabeled data. Clustering and community detection of graph networks used in the previous module, namely hierarchical clustering, modularity maximization, and spectral graph partitioning are unsupervised learning processes. No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself like discovering hidden patterns in data or a means towards an end like feature learning.
# 
# In supervised learning, the objective is to predict features evolutions and set labels to those features corresponding to what we want to achieve or answer through the analysis along with specified objective functions to optimize when fitting to data. The model is made features and corresponding labels and specified objective functions. The data is split into at least two sets: a training (in general with 70% of the data available) set to train the algorithm using a k-fold cross-validation and a test set (in general with the remaining 30%) to evaluate the results obtained by the algorithm and assess the overfitting. Learning or fitting the model means finding optimal parameters of the model structure using provided/available or seen/visible input/target data. This is called training the model and made with the training or fitting dataset called seen or visible data. The second step of the process is to confront the model to the other dataset, the test set kept unseen by the model built in the first phase, in order to assess the general performance of the model. In other words, the model is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.
# 
# In semi-supervised learning, we give an incomplete training signal meaning a training set with some or many of the target outputs missing. Translation is a special case of this principle where the entire set of problem instances is known at learning time, except that part of the targets are missing.
# 
# In reinforcement learning, the model interacts with a dynamic environment in which it must perform a certain goal such as driving a vehicle, playing chess or go or discussing with users online, without a teacher explicitly telling it whether it has come close to its goal.
# 
# Most of machine learning algorithms are called classifier ranging from simple logistic regression to random forests and SVMs. In other words, machine Learning is a type of artificial intelligence (AI) that provides computers with the ability to learn without being explicitly programmed, and ultimately computer programs that can teach themselves to grow and change when exposed to new data. Machine learning identifies patterns using statistical learning and computers by unearthing boundaries in data sets and it can be use to make predictions.
# 
# B. Machine Learning can be used to capturing statistical regularities from systems that are potentially very complex and highly dimensional and make data-driven prediction on future features values in function of attributes and labels of features collected and plugged into the model when information is unavailable or not easily accessible by looking at users directly through their email, cell phone calls or text messages, internet browsing, video apps browsing or credit card logs: location, travel frequency, travel direction, economic, social or demographic characteristics of user like gender, employment status, age, marital status, level of income, social status, hobbies, interests or personality category belongings. And ML can do it with more accuracy than random guess or the gut feeling of functional specialists. And from there for example geographical comparisons are possible to assess cultural differences in behaviors for example. 
# 
# C. Overfitting is overfeeding the model with features that are not relevant. It happens when some boundaries are based on distinctions that don't make a difference, in other words when the model trains and memorizes the data but fails to predict well in the future by over-representing its performance. You can see if a model overfits by having test data flow through the model. There are two styles of general overfitting: over representing performance on particular datasets and (implicitly) over representing performance of a method on future datasets. 
# 
# D. Use regularization and/or cross-validation. Regularization controls the penalty for complexity to prevent under- and over-fitting. And cross-validation separates model selection from testing, resulting in a more conservative estimate of generalization. For the case of over-fitting only, obtaining more training data will also help. To test for under/over-fitting, it is helpful to plot the regression/classification error by some measure of complexity. For under-fitting degrees of complexity, training and testing errors will both be high. For over-fitting degrees of complexity, training errors will be low and testing errors will be high (https://www.quora.com/What-are-ways-to-prevent-over-fitting-your-training-set-data).
# 
# Several types of overfitting are defined in a post titled "Clever Methods of Overfitting" by John Langford, Principal Researcher at Microsoft or Doctor of Learning at Microsoft Research (http://hunch.net/?p=22, https://www.microsoft.com/en-us/research/people/jcl/). In this article, he explains 11 examples:
# 
# . Traditional overfitting: derives from training a complex predictor on too-few examples. To avoid it, select the best data for testing, use a simpler predictor, get more training examples or integrate over many predictors.
# 
# . Parameter tweak overfitting: when you use a learning algorithm with many parameters or choose the parameters based on the test set performance (i.e. the features so as to optimize test set performance). The remedies are the same than in traditional overfitting.
# 
# . Brittle measure overfitting: when you use a brittle measure of performance to overfit (i.e. entropy, mutual information or leave-one-out cross-validation). This is particularly severe when used in conjunction with another approach. To avoid it, prefer less brittle measures of performance. 
# 
# . Bad statistics overfitting: when misused statistics overstate confidences (i.e. using standard confidence intervals on a cross validation of Gaussian independent and identically distributed variables). Remedy: Don’t do this.
# 
# . Choice of measure overfitting: when you choose for the model the best of accuracy, error rate, (A)ROC, F1, percent improvement on the previous best, percent improvement of error rate, (A)ROC, F1. To avoid it, use canonical performance measures directly motivated by the problem. 
# 
# . Incomplete prediction overfitting: instead of making a multiclass prediction, make a set of binary predictions and compute the optimal multiclass prediction. Remedy: don't do it.
# 
# . Human-loop overfitting: when you use a human as part of a learning algorithm and don’t take into account overfitting by the entire human/computer interaction. To avoid it, make sure test examples are not available to the human. 
# 
# . Data set selection overfitting: when you choose to report results on some subset of datasets where your algorithm performs well. Data set selection subverts this and is very difficult to detect. To prevent it, use comparisons on standard datasets and select datasets without using the test set.
# 
# . Reprobleming overfitting: when you alter the problem so that your performance improves. The remedy is simply to make sure problem specifications are clear and avoid doing it. 
# 
# . Old datasets overfitting: when you create an algorithm for the purpose of improving performance on old datasets using for example a process of feedback design indicating better performance than we might expect in the future. To avoid, it, prefer simple algorithm design, weight newer datasets higher in consideration.
# 
# . Overfitting by review: when 10 people submit a review, the one with the best result is accepted. This is a systemic problem very difficult to detect or eliminate. We want to prefer good results, but doing so can result in overfitting. To avoid it, try to be more pessimistic of confidence statements.

# <br>
# <div class="alert alert-info">
# <b>Exercise 1 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 2. Computing behavioral indicators
# ## 2.1 Computing the indicators
# As discussed in the previous notebook, bandicoot allows us analyze and extract behavioral indicators from mobile phone data. With bandicoot, it is easy to load all the users in our Friends and Family dataset and automatically compute their indicators.
# 
# The dataset provided contains 129 users interacting with each other. Each CSV file contains call and text metadata records belonging to a single user.

# #### Load libraries and set options

# In[1]:

# Load libraries and set options.
import matplotlib.pyplot as plt
import bandicoot as bc
from tqdm import tqdm_notebook as tqdm  # Interactive progress bar.
import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import norm
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (15, 9) 
plt.rcParams['axes.titlesize'] = 'large'

## For spectral graph partitioning.
from sklearn.cluster import spectral_clustering as spc

## Supervised learning.
from sklearn import svm, linear_model, ensemble, neighbors, tree
from sklearn import metrics, cross_validation, preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.classifier import Perceptron
from mlxtend.classifier import MultiLayerPerceptron as MLP


# #### 2.1.1 Create a function to load a user and returns all the indicators

# In[2]:

def make_features(user_id):
    '''
    Compute and return indicators of a specific user
    '''
    user = bc.read_csv(user_id, "bandicoot-training-master/data-fnf/records/",
                       attributes_path="../data/bandicoot/attributes/subsector/",
                       describe=False, warnings=False)

    return bc.utils.all(user, summary='extended', split_day=True, split_week=True)


# #### 2.1.2 Create a list of users with their features from all the CSV files 

# In[3]:

# Read all the csv files in the in "./data-fnf/records" directory.
all_features = []

# Process all the files in the specified directory using the function created previously.
for f in glob.glob("./bandicoot-training-master/data-fnf/records/*.csv"):
    user_id = os.path.basename(f)[:-4]  # Remove .csv extension.
    all_features.append(make_features(user_id))
    
# Export all features in one file (fnf_features.csv).
bc.io.to_csv(all_features, 'fnf_features.csv')


# #### 2.1.3 Load the features and attributes to a dataframe

# In[4]:

# Load the features and attributes to a dataframe using the pandas library.
df = pd.read_csv('fnf_features.csv')
df.head()


# ## 2.2 Defining the target variable
# In many situations, it is often that there’s a piece of important information that we are interested that is useful for informing policy initiatives or understanding human behaviour at scale. Particularly for the developing world, the ODI [noted](https://www.odi.org/sites/odi.org.uk/files/odi-assets/publications-opinion-files/9604.pdf):
# 
# >"Data are not just about measuring changes, they also facilitate and catalyse that change. Of course, good quality numbers will not change people’s lives in themselves. But to target the poorest systematically, to lift and keep them out of poverty, even the most willing governments cannot efficiently deliver services if they do not know who those people are, where they live and what they need. Nor do they know where their resources will have the greatest impact"
# 
# In the following example, we have provided an proxy indicator for socio-economic status that can be appended to the family and friends data set. (We could also use other attributes of interests such as age or gender as demonstrated in, for example, [A Study of Age and Gender seen through Mobile Phone Usage Patterns in Mexico](http://arxiv.org/abs/1511.06656)). Our objective is bulding a predictor of socio-economic segment one belongs to using mobile phone metadata. 

# ## 2.3 Exploratory Data Analysis
# 
# Exploratory data analysis (EDA) is a critical first step in any data analysis project and is useful for the following reasons.
# 
# - Detecting errors in data
# - Validating our assumptions
# - Guide in the selection of appropriate models
# - Determining relationships among the explanatory variables
# - Assessing the direction and size (roughly) of relationships between predictor/explanatory and response/target variables
# 
# We have already been using some basic EDA tools (```df.info()``` and ```df.describe()```) that an analyst may wish to use to have a glimpse on the data they will be working with. Other useful EDA approaches including preliminary screening of variables to assess how they relate with the response variable(s). In the following section, we demonstrate how one can explore differences in distributions of features between the groups we are interested in. 

# In[5]:

# First extract the groups by target variable.
sub_gr=df.groupby('attributes__sub')


# ##### Are there any differences in the distribution of 'call durations' between the two groups?

# In[6]:

# Plot both segments.
plt.subplot(2,2,1)
_ = plt.hist(df['call_duration__allweek__allday__call__mean__mean'].dropna(), bins=25, color='green')
plt.title('Both Segments')

# Plot Segment 1.
plt.subplot(2,2,2)
_ = plt.hist(sub_gr.get_group(0) ['call_duration__allweek__allday__call__mean__mean'].dropna().values, bins=25, color='red') 
plt.title('Segment 1')

# Plot Segment 2.
plt.subplot(2,2,3)
_ = plt.hist(sub_gr.get_group(1)['call_duration__allweek__allday__call__mean__mean'].dropna().values,bins=25)
plt.title('Segment 2')


# ##### Are there any differences in the distribution of number of interactions between the two groups?

# In[7]:

# Plot both segments.
plt.subplot(2,2,1)
#df['call_duration__allweek__allday__call__mean__mean'].hist(bins=100)
_ = plt.hist(df['number_of_interaction_in__allweek__night__text__mean'].dropna(), bins=25, color='green')
plt.title('Both Segments')

# Plot segment 1.
plt.subplot(2,2,2)
_ = plt.hist(sub_gr.get_group(0)['number_of_interaction_in__allweek__night__text__mean'].dropna().values, bins=25, color='red')
plt.title('Segment 1')

# Plot segment 2.
plt.subplot(2,2,3)
_ = plt.hist(sub_gr.get_group(1)['number_of_interaction_in__allweek__night__text__mean'].dropna().values,bins=25)
plt.title('Segment 2')


# ## 2.4 Data preprocessing

# We need to preprocess the data to make it usable for machine learning purposes. This involves a number of activities such as 
# 1. Assigning numerical values to factors
# 1. Handling missing values
# 2. Normalize the features (so features on small scales do not dominate when fitting a model to the data)
# 
# > **Note**: In order to achieve that and to have remaining data we impute the missing values. In other words, you make an educated guess about the missing data.
# 
# Sklearn fuctions:
# - `preprocessing`
# - `preprocessing.Imputer()`
# 
# > **Note**: Small scale imply big value. As example, assume we are classifying people based on their physiological characteristics. A small scale feature would be weight measured in kilograms vs height measured in meters. 
# 
# Sklearn functions:
# - `preprocessing.StandardScaler()`

# First, make sure that we only have records in the dataset that are labeled. Our target variable is a socio-economic binary indicator   ```df.attributes__sub```

# In[8]:

# Drop records with missing labels in the target variable.
df = df[~df.attributes__sub.isnull()]


# We create two objects to use in building the classifier:
# 
# - the array ``y`` contains the labels we want to predict (cluster 1 / cluster 2),
# - the matrix ``X`` contains the features for all users (one column for one feature, one line for one user).

# In[9]:

# 1. Target variable.
y = df.attributes__sub.astype(np.int)
y.values


# > **Note**: The cell below will produce a warning message stating that "A value is trying to be set on a copy of a slice from a DataFrame" which you can ignore.

# In[10]:

# Convert gender labels to binary values (zero or one):
df.loc[:,'attributes__gender'] = (df.attributes__gender == 'female').values.astype(np.int)


# In[11]:

# 2. Remove columns with reporting variables and attributes. (the first 39 and the last 2):
df = pd.concat([df[df.columns[39:-5]],df[df.columns[-3]]], axis=1)
X = df.values
X


# In[12]:

# 3. Impute the missing values in the features.
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)


# #### Normalization
# Normalize the data to center it around zero and transform it to a similiar scale to prevent variables in 'small' units (and therefore high values) to dominate the classification unreasonably.

# In[13]:

# 4. Preprocess data (center around 0 and scale to remove the variance).
scaler = preprocessing.StandardScaler()
Xs = scaler.fit_transform(X)


# ## 2.5 Classification with cross-validation
# 
# > Support vector machines (SVMs) one of the popular linear classifiers that are based on two central ideas of margin maximization and kernel functions. Margin maximization allows one to find an optimal hyperplane that separates linearly data, while kernel functions extend the algorithm to handle data not linearly separable by mappping the data into a new high-dimensional space. A rough intuitive explanation of how it works can be found here: http://www.dataschool.io/comparing-supervised-learning-algorithms/

# As we have seen seen in the video and explained above, splitting the data into test and training sets is critical to avoid overfitting and, therefore, generalize to real previously unseen data. Cross-validation extends this idea further. Instead of having a single train/test split, we can specify so-called folds such that our data is divided into similarly sized folds. Training occurs by taking all folds except 1, also referred to as the holdout sample. On completion of training, we test the performance of our fitted model using the holdout sample. The holdout sample is then thrown back with the rest of the other folds, a different fold pulled out as the new holdout sample. Training is repeated again with the remaining folds and we measure performance using the holdout sample. This process is repeated until each fold had a chance to be a test or holdout sample. The expected performance of the classifier, called cross-validation error, is then simply an average of error rates computed on each holdout sample. We demonstrate this process first by performing a standard train/test split and then computing cross-validation error.

# In[14]:

# 5. Divide records in training and testing sets.
np.random.seed(2)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xs, y, test_size=0.3, stratify=y.values)

# 6. Create an SVM classifier and train it on 70% of the data set.
clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

# 7. Analyze accuracy of predictions on 30% of the holdout test sample.
classifier_score = clf.score(X_test, y_test)
print '\nThe classifier accuracy score is {:.2f}\n'.format(classifier_score)


# To get a better measure of prediction accuracy (which we can use as a proxy for goodness of fit of the model), we can successively split the data in folds that we use for training and testing:

# In[15]:

# Get average of 3-fold cross-validation score using an SVC estimator.
n_folds = 3
cv_error = np.average(cross_val_score(SVC(), Xs, y, cv=n_folds))
print '\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error)


# ## 2.6 Receiver operating characteristic (ROC) curve. 
# 
# In statistical modeling and machine learning,  a commonly reported performance measure of model accuracy is Area Under the Curve (AUC), where by *curve* the ROC curve is implied. ROC stands for *Receiver Operating Characteristic* - a term originated from the second world war and used by by radar engineers.
# 
# To understand what information the ROC curve conveys, consider the the so-called confusion matrix that basically is a 2-dimensional table where the classifier model is on one axis (vertical) and ground truth is on the other (horizontal) axis as shown below. Either of these axis can take two values as depicted. A cell in the table then is an intersection where the conditions on each the dimensions hold. For example, in the top left cell, the model condition is "A" and the ground truth is also "A". Hence, the count of instances where these 2 conditions are true (for a specific data point) is captured, hence the label 'True positive'. The same logic applies to the rest of the other cells. The total of the counts in these cells therefore must equal the number of data instances in our data set under consideration.
# 
# 
# ~~~~
#                         Actual: A        Not A
# 
#   Model says “A”       True positive   |   False positive
#                       ----------------------------------
#   Model says “Not A”   False negative  |    True negative
#   
# ~~~~
# 
# 
# 
# In an ROC curve, we plot ‘True Positive Rate‘ on Y-axis and ‘False Positive Rate‘ on the X-axis, where the the “true positive”, “false negative”, “false positive” and “true negative” are events (or their probability) as described above, and where the rates are defined according to:
# 
# > True positive rate (or sensitivity)}:  **tpr = tp / (tp + fn)**
# 
# > False positive rate:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                   **fpr = fp / (fp + tn)**
# 
# > True negative rate (or specificity):   **tnr = tn / (fp + tn)**
# 
# In all definitions the column total is the denominator. We can therefore express the true positive rate (tpr) as the probability that the model says “A” when the real value is indeed A (i.e., a conditional probability). This does not tell you how likely you are to be correct when calling “A” (i.e., the probability of a true positive, conditioned on the test result being “A”).
# 
# To interpret the ROC correctly, consider the points that lie on along the diagonal represent. For these situation, there is an equal chance of "A" and "not A" happening. Therefore,  this is not that different from making a prediction by tossing of an unbiased coin, or put simply, the classification model is random.
# 
# For points above the diagonal, **tpr** > **fpr**, and our model says we are in a zone where we are performing better than random. For example, assume **tpr ** = 0.6 and **fpr** = 0.2, then the probability of our being in the true positive group is $(0.6 / (0.6 + 0.2)) = 75\%$. Furthermore, holding **fpr** constant, it is easy to see that the more vertically above the diagonal we are positioned, the better the classification model. Further basic details on the correct interpretation can be found in this [reference](\http://pubs.rsna.org/doi/pdf/10.1148/radiographics.12.6.1439017).

# In[16]:

# The confusion matrix helps visualize the performance of the algorithm.
y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = metrics.confusion_matrix(y_test.values, y_pred)

print(cm)


# In[17]:

# Plot the receiver operating characteristic curve (ROC).
plt.figure(figsize=(20,10))
probas_ = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.axes().set_aspect(1)


# <br>
# <div class="alert alert-info">
# <b>Exercise 2 Start.</b>
# </div>
# 
# ### Instructions
#  > What does the score and ROC curve tell us about our classifier and how it compares to "flipping an unbiased coin" (random) to determine the class to which an observed data point belongs?
#  
#  **Hint**
#  > See more details regarding the ROC and its interpretation [here](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
# 
# Provide your answer in a markdown cell below.

# The Receiver Operating Characteristic (ROC) curve plots ‘True Positive Rate‘ on Y/ordinate/horizontal axis and ‘False Positive Rate‘ on the X/abscissa/vertical axis, where the the “true positive”, “false negative”, “false positive” and “true negative” are probabilities of events. In other words, the ROC curve, is a graphical plot that illustrates the performance of a binary classifier system as its discrimination threshold is varied. 
# 
# The true-positive rate (TPR) is also known as sensitivity, recall or probability of detection. The false-positive rate (FPR) is also known as the fall-out or probability of false alarm. Thus, the ROC curve is the sensitivity as a function of fall-out. If the probability distributions for both detection and false alarm are known, the ROC curve can be generated by plotting the cumulative distribution function of the detection probability in the y-axis (Area Under the probability distribution from -infinity to the discrimination threshold) versus the cumulative distribution function of the false-alarm probability in x-axis. As use cases, ROC curves give a graphic representation to assess or diagnostic the accuracy of a test, choose the most optimal cut-off of a test or compare accuracy of several tests.
# 
# As described above, to interpret the ROC correctly, we must consider the points that lie on along the diagonal of the graph and when the ROC curve touch or go under the diagonal, there is an equal chance for the model result or event to happen and not happen, corresponding to a point where the probability of happening of the event predicted by the model is similar to the random tossing of an unbiased coin, or in other words that the classification model is random. The perfect classification would be at the extreme top left of the graph and the curve above the diagonal (TPR > FPR) shows probabilities of predictions of the model better than a random process (above 50%) and below worse (below 50%). The machine learning community most often uses the ROC AUC statistic for model comparison, even if the AUC in that use is now controversial for its natural high noise as classification measure.
# 
# In this example above, our classifier system seems to perform well at the exception a point at TPR of approximately 0.25 and FPR of approximately 0.3 where the probability of our result being in the true positive group is only (0.25/(0.25+0.3))=45%.

# <br>
# <div class="alert alert-info">
# <b>Exercise 2 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# <br>
# <div class="alert alert-info">
# <b>Exercise 3 Start.</b>
# </div>
# 
# ### Instructions
#  > 1. How many data points in the test set from class 1 were predicted as class 1 by the trained classifier?
#  > 2. How many data points in the test set from class 1 were predicted as class 0 by the trained classifier?
#  > 3. How many data points in the test set from class 0 were predicted as class 0 by the trained classifier?
#  > 4. How many data points in the test set from class 0 were predicted as class 1 by the trained classifier?
#  > 5. What is the error rate on the test set for this classifier?
#  > 6. Assuming class 0 as the positive class, can you calculate the true positive rate or sensitivity of this classifier?
#  > 7. What is the specificity assuming class 1 is the negative class?
#  
#  **Hints**
#  > Descriptions and formulae for **sensitivity** and **specificity** can be found [here](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
# 
# 

# The confusion matrix or CM is:
# [[ 9  9]
#  [ 6 15]] 
#  
# which means TP = 9 FP = 9 FN = 6 and TN = 15.
# 
# True positive rate (TPR or sensitivity) = TP/(TP+FN) = 9/(9+6) = 60%
# 
# False positive rate (FPR) = FP/(FP+TN) = 9/(9+15) = 37.5%
# 
# True negative rate (TNR) or specificity (SPC) = TN/(FP+TN) = 15/(9+15) = 62.5%
# 
# ACC = (TP+TN)/(TP+FP+FN+TN) = (9+15)/(9+9+6+15) = 24/39 = 61.5%
#  
# A. There is 9 data points where the test set from class 1 were predicted as class 1 by the trained classifier
# 
# B. There is 9 data points in the test set from class 1 were predicted as class 0 by the trained classifier
# 
# C. There is 6 data points in the test set from class 0 were predicted as class 0 by the trained classifier
# 
# D. There is 15 data points in the test set from class 0 were predicted as class 1 by the trained classifier
# 
# E. The error rate (ER) = (FP+FN)/(TP+FP+FN+TN) = (9+6)/39 = 38.5%
# 
# F. If class 0 is the positive class, I presume thath the matrix is reversed from TP FP FN TN to FP TP TN FN with FP = 
# 9 TP = 9 TN = 6 FN = 15 and the sensitivity or TPR = TP/(TP+FN) = 9/(9+15) = 37.5%, FPR = 9/(9+6) = 60% and 
# 
# G. TNR or specificity = 6/(9+6) = 40% 

# <br>
# <div class="alert alert-info">
# <b>Exercise 3 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 3. Optimal classifier parameters

# The examples above made use of the SVC() function using default parameters. Usually, you would want to optimize the setting of these parameters for a given problem, as these are learned by the algorithm during the training phase. In the case of support vector classifiers, these parameters include kernel choice,
# the kernel parameters (Gaussian kernel: $\gamma$; Polynomial kernel: $d$), as well as the penalty for misclassification ($C$). For an illustration of the behavior of these and other kernels explore the [scikit docs](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)resource. 
# 
# Tuning parameters for an optimal solution is inherently difficult. A popular approach is to perform a search over the a grid defined across the various parameters to be optimized for. The grid search function is illustrated next. This illustration will consider optimizing over two parameters - $C$ (misclassification cost) $\gamma$ (The RBF kernel parameter).

# In[18]:

# Train classifiers.
param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)


# In[19]:

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# In[20]:

grid.best_estimator_.probability = True
clf = grid.best_estimator_


# In[21]:

y_pred = clf.fit(X_train, y_train).predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

print(cm)


# ## 3.1 Using other classifiers
# 
# The SVM classifier used is just one option of classifiers that you have at your disposal. There are other classification methods implemented in scikit-learn (as well as mlxtend) that you can easily use. These include:
# 
# - Decision trees with ``tree.DecisionTreeClassifier()``;
# - K-nearest neighbors with ``neighbors.KNeighborsClassifier()``;
# - Random forests with ``ensemble.RandomForestClassifier()``;
# - Perceptron (both gradient and stochastic gradient) ``mlxtend.classifier.Perceptron``; and 
# - Multilayer perceptron network (both gradient and stochastic gradient) ``mlxtend.classifier.MultiLayerPerceptron``.
# 
# It is important to understand the underlying technique, as well as the implementation, in order to correctly interpret the output, or tune the estimator parameters. Next, the use of some of these classifiers is illustrated on the data set using the above-mentioned libraries.

# In[22]:

# Create an instance of random forest classifier, fit the data, and assess performance on test data.
clf_rf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,max_depth=5 )    
n_folds = 3
cv_error = np.average(cross_val_score(clf_rf, 
                                      X_train, 
                                      y_train, 
                                      cv=n_folds))
clf_rf.fit(X_train, y_train)
print '\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error)                             


# In[23]:

# Create an instance of logistic regression classifier, fit the data, and assess performance on test data.
clf_logreg = linear_model.LogisticRegression(C=1e5)    
n_folds = 3
cv_error = np.average(cross_val_score(clf_logreg, 
                                      X_train, 
                                      y_train, 
                                      cv=n_folds))
clf_logreg.fit(X_train, y_train)
print '\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error) 


# In[24]:

# Create an instance of decision tree classifier, fit the data, and assess performance on test data.
clf_tree = tree.DecisionTreeClassifier()
n_folds = 3
cv_error = np.average(cross_val_score(clf_tree, 
                                      X_train, 
                                      y_train, 
                                      cv=n_folds))
clf_tree = clf_tree.fit(X_train, y_train)
print '\nThe {}-fold cross-validation accuracy score for this classifier is {:.2f}\n'.format(n_folds, cv_error)


# In[25]:

# Create an instance of multilayer perceptron classifier (gradient descent), fit the data, and assess performance on test data.
clf_nn1 = MLP(hidden_layers=[40],l2=0.00,l1=0.0,epochs=150,eta=0.05,momentum=0.1,decrease_const=0.0,minibatches=1,random_seed=1,print_progress=3)
clf_nn1 = clf_nn1.fit(X_train, y_train)
clf_nn1.score(X_test, y_test)


# In[26]:

# Create an instance of multilayer perceptron classifier (stochastic gradient descent), fit the data, and assess performance on test data.
clf_nn2 = MLP(hidden_layers=[40],l2=0.00,l1=0.0,epochs=50,eta=0.05,momentum=0.1,decrease_const=0.0,minibatches=len(y_train),random_seed=1,print_progress=3)
clf_nn2 = clf_nn2.fit(X_train, y_train)
clf_nn2.score(X_test, y_test)


# In[27]:

# Plot the results.
colors = ['b', 'g', 'r','c','m','k','y']
classifiers = ['svm','random_forest', 'logistic regression', 'decision tree', 'mlp_gd', 'mlp_sgd']
plt.figure(figsize=(20,10))
for i, cl in enumerate([clf, clf_rf, clf_logreg, clf_tree,clf_nn1, clf_nn2]):
    probas_ = cl.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label=classifiers[i]+' (AUC = %0.2f)' % (roc_auc))
    
plt.plot([0, 1], [0, 1], '--', color=colors[i], label='Random (AUC = 0.50)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])   
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.axes().set_aspect(1)
plt.legend(loc="lower right")


# <br>
# <div class="alert alert-info">
# <b>Exercise 4 Start.</b>
# </div>
# 
# ### Instructions
#  > 1. Provide some general comments on the performance of the different classifiers based on the ROC plots.
#  > 2. List two limitations you can think of that one should be aware of when using ROC curves to assess classifier performance.
#  > 3. List two ways to improve the performance of the models.
#     
#  **Hints**
#  > As part of your general comments (requested in 1), rank the different classifiers based on the overall performance. Also, include comments on the relative performance of each classifier as the false positive rate is increased. Lastly, consider whether the use of an ensemble of these classifiers based on majority vote could improve the observed performance of the best classifier (general comments).
#  
#  >*Majority voting refers to using multiple classifiers for decision making, and returning as output the class which is in the majority. In other words, each classifier's output is considered a vote and the output that has the most votes is the winner.*

# A. General comments:
# 
# Only 2 models are significantly underperforming: logistic regression and decision tree. The area under the ROC curve is a summary statistic that indicates higher overall performance with higher AUC. Ranked from higher to lower AUC or overall performance we have 1. mlp_sgd 2. mlp_gd 3. svm 4. random_forest 5. logistic regression 6. decision tree. As the false positive rate increase the performance of all models improve, with the random forest, SVM and mlp_gd classifiers catching up on the mlp_sgd performance. SVM and random forest start on low FPR by underperforming with random forest being the worse, but their performance improve significantly at a level of 0.4 FPR. A model combining multilayer perceptron classifier (stochastic gradient descent and gradient descent), SVM and random forest would have a better overall performance across the board. With such a combined model, the over performance of the multilayer perceptron classifiers at low FRP balancing the underperformance of the SVM and random forest at these levels and at higher FRP, the SVM and random forest would accelerate the over performance of the model.
# 
# B. Limitation of ROC analysis, examples:
# 
# . One of the major limitations of ROC analysis is that data must be divided into two states. This raises another problem of whether the data will clearly fall into one state or the other. In some cases, the frontier is blurred.
# 
# . Another limitation is that ROC AUC treats sensitivity and specificity as equally important overall when averaged across all thresholds. But it is always the case in light of the study objective/goal, question or problem/issue studied? 
# 
# . Besides, confidence scales used to construct ROC may be inconsistent and unreliable for the study objective/goal, question or problem/issue studied. 
# 
# . A fourth limitation, as shown in the article of Steve Halligan, Douglas G. Altman, and Susan Mallett titled "Disadvantages of using the area under the receiver operating characteristic curve to assess imaging tests: A discussion and proposal for an alternative approach" (http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4356897/)
# would be the prevalence of abnormality since AUC is unchanged at differing prevalence. In clinical practice for example, "the number of patients classified accurately by a test changes with prevalence. In high-prevalence situations the number of test-positive patients increases greatly for a given increase in sensitivity compared with low-prevalence situations (e.g., screening). AUC itself cannot account for how changing prevalence impacts on results for individual patients, so instead sensitivity and specificity are used, with the operating point directed by prevalence. While sensitivity and specificity are prevalence-independent, these measures separate positive and negative patients so prevalence can be incorporated by users as part of their interpretation."
# 
# C. Ways to improve the performance of the models
# 
# According to Sunil Ray in "8 Proven Ways for improving the “Accuracy” of a Machine Learning Model" in Analytics Vidhya (https://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/) there is 8 proven ways to re-structure your model approach to improve the predictive power of models. In our case, beside combining them through Bagging (Bootstrap Aggregating) or Boosting as suggested in the general comments, we could add more data, add or transform features (feature engineering) to increase the explanatory power of the model, or in the same manner, select the best features only, with high explanatory power, in other words which better explains the relationship of independent variables with target/response variable. Another precaution would be to check again the treatment of missing or outlier values and adjust if necessary or fine tune the algorithm by parameter tuning to find the optimum value for each parameter. 

# <div class="alert alert-info">
# <b>Exercise 4 End.</b>
# </div>
# 
# > **Exercise complete**:
#     
# > This is a good time to "Save and Checkpoint".

# # 4. Submit your notebook
# 
# Please make sure that you:
# - Perform a final "Save and Checkpoint";
# - Download a copy of the notebook in ".ipynb" format to your local machine using "File", "Download as", and "IPython Notebook (.ipynb)"; and
# - Submit a copy of this file to the online campus.

# # 5. References
# Scikit-learn. 2010-2014. “1.4 Support Vector Machines.” Accessed August 16. http://scikit-learn.org/stable/modules/svm.html. 

# In[ ]:



