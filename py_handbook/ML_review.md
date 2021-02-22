<center><span style="font-size:42px;color:#8B30BB;">Machine Learning Review:  </center></span>

`Siming Yan` (@www.fyenneyenn.studio)

`01,2021`

`According to: Kaggle, ISLR, Matt:Bussiness data science, AAE722 machine learning in R  `

---

# Kaggle: **Decision tree**

* both regression and classification
* top-down, greedy approach that is known as recursive binary splitting. It is greedy because at each step of the tree-building process, the **best split is made at that particular step** 
* a better strategy is to grow a very large tree T0, and then prune it back in order to obtain a subtree to avoid overfitting problem.
* non robust and low accuracy.

```python
from sklearn.ensemble import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# Define model
melbourne_model = DecisionTreeRegressor(random_state = 1)
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```

Many machine learning models allow some randomness in model training. Specifying a number for `random_state` (`setseed()`)ensures you get the same results in each run.

```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

---

You can see in scikit-learn's [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) that the decision tree model has many options (more than you'll want or need for a long time). The most important options determine the ***tree's depth***. 

When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).

This is a phenomenon called **overfitting**, where a model matches the training data almost perfectly, but does poorly in validation and other new data. On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.

## Traning model

<img src="/Users/fyenne/Downloads/booooks/semester5/pythonAAE/py_handbook/pic_for_md/overfitting.png" alt="overfitting" style="zoom:67%;" />

 We want the lowest validation set MAE. Following is the MAE 调参方程

```python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

调参 for loop.制作dataframe 用以存储MAE 值和nodes值

```python
summ = pd.DataFrame(candidate_max_leaf_nodes) # dataframe with leaf nodes values 
l = [] # empty list for further append.

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 750]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    l.append(my_mae) # create a list to store the values of my_mae, so that its readable 
    print(my_mae)

summ['mae_list'] = l
best_tree_size = int(summ[summ.mae_list == summ.mae_list.min()][0])

```

complete training and find the final best leaf_nodes number' 

## Fill'in best parameters and re-train the model.

```python
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size,random_state = 1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)
```

---

# Kaggle Random Forrests. Scikit Learning.



```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```

```python
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1)

# fit your model
rf_model.fit(train_X, train_y)
val_predictions = rf_model.predict(val_X) #这里注意要fit然后也要predict的，用的是唯一一个模型。

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

# Validation MAE for Random Forest Model: 21857.15912981083
#--------------------------------------------
# for decision trees:
# Validation MAE when not specifying max_leaf_nodes: 29,653
# Validation MAE for best value of max_leaf_nodes: 27,283
```

## write a random forrest Parameter choosing function:

```python
def get_mae_rf(n_estimators, #max_depth,  # para s
               train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=n_estimators, # max_depth = max_depth, 
                                  random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

```python
l = []
candidate_n_estimators = [5,10, 15, 25, 50, 100, 150]
summ = pd.DataFrame(candidate_n_estimators)
# candidate_max_depth = [50, 100, 150, 200, 250]

for n_estimators in candidate_n_estimators:
    my_mae = get_mae_rf(n_estimators, train_X, val_X, train_y, val_y)
    l.append(my_mae)
    print(my_mae)
    #this is a list of mae when choosing parameters.

```

```python
summ['mae_list'] = l
best_n_estimators = int(summ[summ.mae_list == summ.mae_list.min()][0])
best_n_estimators
# best num of estimators. 25
```

## easy_MAE check with RF

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0) #25 should be best
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```

## tuning the hyperparameters

### grid search: SVM

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [
  {'kernel': ['rbf','sigmoid', 'linear'], 'gamma': [1e-3, 1e-4, 1e-2], 'C': [1, 10, 100, 1000]}]
  
# Objective metrics
scores = ['precision']


clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % scores[0])
clf.fit(X_train, y_train)

print("Best Hyperparameters found are:")
print(clf.best_params_)

print("Grid scores are:")

means = clf.cv_results_['mean_test_score']
for mean,params in zip(means, clf.cv_results_['params']):
  print("%0.3f for %r" % (mean, params))

```

### Random Search: RF

```python
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# Download the MNIST Dataset - using load_digits funct
digits = load_digits()
X, y = digits.data, digits.target

# build a random forest classifier
rfc = RandomForestClassifier(n_estimators=20)

# specify parameters and distributions to sample from
hp = {"max_depth": [4, 3, 2],
              "min_samples_split": sp_randint(2, 11),
              "criterion": ["gini", "entropy"],
              "bootstrap": [True, False]
              }

# run randomized search
n_iter_search = 10
random_search = RandomizedSearchCV(rfc, param_distributions=hp,
                                   n_iter=n_iter_search, cv=5, iid=False)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
means = random_search.cv_results_['mean_test_score']
stds = random_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
  print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


```





### Bayesian Optimization: Tensorflow & nn

```python
import warnings
warnings.filterwarnings("ignore")

#Importing liberaries
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam


from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import numpy as np

# Define the Hyperparameters ranges:
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],name='activation')


HPs = [dim_learning_rate,dim_num_dense_layers,dim_num_dense_nodes,dim_activation]


#It is a good idea to define a default range.
default_parameters = [1e-5, 1, 16, 'relu']

#Import the MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)
validation_data = (data.validation.images, data.validation.labels)


#Define the data dimension
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    
    # Start construction of a Keras Sequential model.
    model = Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(InputLayer(input_shape=(img_size_flat,)))

    # The input from MNIST is a flattened array with 784 elements,
    # but the convolutional layers expect images with shape (28, 28, 1)
    model.add(Reshape(img_shape_full))

    # First convolutional layer.
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation=activation, name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Second convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation=activation, name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))
    
    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

@use_named_args(dimensions=HPs)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)

    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
   
    # Use Keras to train the model.
    history = model.fit(x=data.train.images,
                        y=data.train.labels,
                        epochs=1,
                        batch_size=128,
                        validation_data=validation_data)

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

search_result = gp_minimize(func=fitness,
                            dimensions=HPs,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

                      
```



# To clean with *<span style="font-size:32px;color:#8B30BB;">values</span>*.

## check NAs.

```python
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

<center><span style="font-size:42px;color:#8B30BB;">Machine Learning Review:  </center></span>
```

### drop NAs

```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
#--------------------------------------------
# the same results of creating a list of missing value cols.
l = []
for col in X_train.columns:
    if X_train[col].isnull().any():
        l.append(col)
#--------------------------------------------
        
# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```



### Imputation

```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) 
# two different method of imputing. above is fitting, below is mean()
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
#--------------------------------------------
# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```



### An Extension to Imputation

```python
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```

~~Given that thre are so few missing values in the dataset, we'd expect imputation to perform better than dropping columns entirely.~~**However, we see that dropping columns performs slightly better!** While this can probably partially be attributed to **noise** in the dataset, ~~another potential explanation is that the imputation method is not a great match to this dataset.~~That is, maybe instead of filling in the mean value, it makes more sense to set every missing value to **a value of 0,** to fill in the most frequently encountered value, or to use some other method. For instance, consider the `GarageYrBlt` column (which indicates the year that the garage was built). It's likely that in some cases, a missing value could indicate a house that does not have a garage. Does it make more sense to fill in the median value along each column in this case? Or could we get better results by filling in the minimum value along each column? It's not quite clear what's best in this case, but perhaps we can rule out some options immediately - for instance, setting missing values in this column to 0 is likely to yield horrible results!

## Categorical variables.

```python
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```



### drop:

```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```



### label encoding:

```python
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
# 标签是（int）数据形的，所以可以用fit_transform. 而在Valid中，数据更敏感，不适用fit而使用普通Transform

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```

In the code cell above, for each column, we randomly assign each unique value to a different integer. This is a common approach that is simpler than providing custom labels; however, we can expect an additional boost in performance if we provide better-informed labels for all ordinal variables.

### one-hot encoding (`R: fastDummies`)

```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1) 
# here should be all categorical columns removed.
num_X_valid = X_valid.drop(object_cols, axis=1) #

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```

- We set `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data, and

- setting `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

  ---

  

  **how many categories in each category variable**

```python
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

```

### standard scaling:

```python
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include = np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include = object)),
)
```

## further Data cleaning: erase non-frequent data:

`Storing the non-rare labels (labels > 5%) in a csv file would help in test set preparation.`

```python
non_rare = pd.DataFrame()
for i in categorical_features:
    var_dist = data[i].value_counts().copy()
    var_dist = (var_dist / var_dist.sum()).copy()
    non_rare = pd.concat([non_rare,pd.DataFrame({i:var_dist[var_dist>0.05].index})],axis=1).copy()

non_rare.to_csv('./non_rare_categories.csv',index=False)
```



---

# piplines

**Pipelines** are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

Many data scientists hack together models without pipelines, but pipelines have some important benefits. Those include:

1. **Cleaner Code:** Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step.

2. **Fewer Bugs:** There are fewer opportunities to misapply a step or forget a preprocessing step.

3. **Easier to Productionize:** It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go into the many related concerns here, but pipelines can help.

4. **More Options for Model Validation:** You will see an example in the next tutorial, which covers cross-validation.

   

## Define Preprocessing Steps

preprocessing functions -> models set up -> pipline combination

---

### prepare preprocessing functions

```python

categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]


```

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

```python
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```



# Cross validation

In **cross-validation**, we run our modeling process on different subsets of the data to get multiple measures of model quality.

For example, we could begin by **dividing the data into 5 pieces**, each 20% of the full dataset. In this case, we say that we have broken the data into 5 "**folds**".

- In **Experiment 1**, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.
- In **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
- We repeat this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).

So, given these tradeoffs, when should you use each approach?

- *For small datasets*, where extra computational burden isn't a big deal, you should run cross-validation.
- *For larger datasets*, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
```

```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):")
print(scores.mean())
```

```python
# easier way:
```

```python
def get_score(n_estimators):
    #"""Return the average MAE over 3 CV folds of random forest model.
    my_pipeline = Pipeline(steps = [
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state = 0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
    return scores.mean()

```

# Kaggle XGB

 

**Gradient boosting** is a method that goes through cycles to iteratively add models into an ensemble.

- First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
- These predictions are used to calculate a loss function (like [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), for instance).
- Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (*Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) on the loss function to determine the parameters in this new model.*)
- Finally, we add the new model to ensemble, and ...
- ... repeat!

```python
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train) # naive ensemble
```

```python
from sklearn.metrics import mean_absolute_error
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
```

`n_estimators` specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.

```python
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)
```

> `early_stopping_rounds`

`early_stopping_rounds` offers a way to automatically find the ideal value for `n_estimators`. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`. It's smart to set a high value for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating. 

> `learning_rate`

Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.

This means each tree we add to the ensemble helps us less. So, we can set a higher value for `n_estimators` without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.

In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets `learning_rate=0.1`.

> `n_jobs`

On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on your machine. On smaller datasets, this won't help.

The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the `fit` command.

Here's the modified example:

 ```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
              # stop after 5 straight rounds of deteriorating validation scores.
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
 ```

> Pretty good prediction accuracy.

# data leakage

**Data leakage** (or **leakage**) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.

```python
# not clear yet what are these functions for:
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
```



```python
#Cross-val accuracy
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
## 0.828650 
```

we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model would likely do much worse than that (in spite of its higher apparent score in cross-validation).

https://www.kaggle.com/fyenneyenn/exercise-data-leakage/edit

---

耐克问题：工厂本月毛皮使用量, 在预测鞋带生产上，是perfect  indicator 还是data leakage?

```Nike has hired you as a data science consultant to help them save money on shoe materials. Your first assignment is to review a model one of their employees built to predict how many shoelaces they'll need each month. The features going into the machine learning model include:

```

- The current month (January, February, etc)
- Advertising expenditures in the previous month
- Various macroeconomic features (like the unemployment rate) as of the beginning of the current month
- The amount of leather they ended up using in the current month

```The results show the model is almost perfectly accurate if you include the feature about how much leather they used. But it is only moderately accurate if you leave that feature out. You realize this is because the amount of leather they use is a perfect indicator of how many shoes they produce, which in turn tells you how many shoelaces they need.

```

This is tricky, and it depends on details of how data is collected (which is common when thinking about leakage). Would you at the beginning of the month decide how much leather will be used that month? If so, this is ok. But if that is determined during the month, you would not have access to it when you make the prediction. If you have a guess at the beginning of the month, and it is subsequently changed during the month, the actual amount used during the month cannot be used as a feature (because it causes leakage).



<center><span style="font-size:42px;color:#8B30BB;">TensorFlow! Machine Learning Review:</center></span>



# Setup tensorflow on Mac M1 chip

create virtual environment: 

`download miniconda.`

`conda create name --=-- python = 3.8.6 (maybe, I forgot.)`

Basically it follows this aritcle.

`https://zhuanlan.zhihu.com/p/343769603`

yeah I dont remember how it worked.

```python
tf.__version__
# '2.4.0-rc0'
# you got this if ur right!
```



# basic shit of tensorflow

```html
https://tf.wiki/zh_hans/basic/basic.html
kaggle
https://www.tensorflow.org/api_docs/python/tf/data/Dataset
https://www.tutorialspoint.com/tensorflow/tensorflow_basics.htm
```

```python
import tensorflow as tf
import numpy as np

matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype = 'int32')
matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype = 'int32')

print (matrix1)
print (matrix2)
# tensorflow matrix calculations !
matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)
# ---
matrix_product = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1,matrix2)
matrix_3 = np.array([(2,7,2),(1,4,2),(9,0,2)],dtype = 'float32')
#--------------------------------------------
matrix_det = tf.matrix_determinant(matrix_3)
with tf.Session() as sess:
   result1 = sess.run(matrix_product)
   result2 = sess.run(matrix_sum)
   result3 = sess.run(matrix_det)
print (result1)
print (result2)
print (result3)
```

```python
# 生成随机数 [1，3]
tf.random.uniform(shape = (1,3))
# 0
tf.zeros(shape=(2))
# matrix
A = tf.constant([[1., 2.], [3., 4.]])

```



## 自动求导

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)
# x ==> variable with an initial value
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x) + x
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)
```

## linear regression

```python
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
# 归一化
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
#--------------------------------------------
# 求解
X = tf.constant(X)
y = tf.constant(y)
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]
num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

a, b # solved # zip() 函数是 Python 的内置函数。用自然语言描述这个函数的功能很绕口，但如果举个例子就很容易理解了：如果 a = [1, 3, 5]， b = [2, 4, 6]，那么 zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]

```

## TensorFlow 模型建立与训练

+ 模型的构建： tf.keras.Model 和 tf.keras.layers

* 模型的损失函数： tf.keras.losses

* 模型的优化器： tf.keras.optimizer

* 模型的评估： tf.keras.metrics

```python
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
    def call(self, input):
        output = self.dense(input)
        return output
# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      
        # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    
    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
```

# Stochastic Gradient Descent

We've described the problem we want the network to solve, but now we need to say how to solve it. This is the job of the optimizer. The optimizer is an algorithm that adjusts the weights to minimize the loss.

Virtually all of the optimization algorithms used in deep learning belong to a family called stochastic gradient descent. They are iterative algorithms that train a network in steps. One step of training goes like this:

Sample some training data and run it through the network to make predictions.

* Measure the loss between the predictions and the true values.

* Finally, adjust the weights in a direction that makes the loss smaller.

* Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)

Each iteration's sample of training data is called a **minibatch** (or often just "batch"), while a complete round of the training data is called an epoch. The number of **epochs** you train for is how many times the network will see each training example.

The animation shows the linear model from Lesson 1 being trained with SGD. The pale red dots depict the entire training set, while the solid red dots are the minibatches. Every time SGD sees a new minibatch, it will shift the weights (w the slope and b the y-intercept) toward their correct values on that batch. Batch after batch, the line eventually converges to its best fit. You can see that the loss gets smaller as the weights get closer to their true values.

Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way). The size of these shifts is determined by the learning rate. A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.

The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds. Their interaction is often subtle and the right choice for these parameters isn't always obvious. (We'll explore these effects in the exercise.)

Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results. Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.



```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

features_num  # numerical variabels. 
features_cat # categorical variables
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)
#--------------------------------------------
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100
y_valid = y_valid / 100

input_shape = [X_train.shape[2]]
#--------------------------------------------
# build up model.
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mae',
)
#--------------------------------------------
# run model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
  	verbose = 0, # ?
)
#--------------------------------------------
# plot loss function descending
# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();

```

# Add `early stopping` to avoid `overfitting`:



overfitting:

<img src="/Users/fyenne/Downloads/booooks/semester5/pythonAAE/py_handbook/pic_for_md/overfitting.png" alt="overfitting" style="zoom:67%;" />

<img src="/Users/fyenne/Downloads/booooks/semester5/pythonAAE/py_handbook/pic_for_md/eUF6mfo.png" alt="eUF6mfo" style="zoom:80%;" />

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

... # define model
... # model compile 

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
```

# Dropout and Batch Normalization

In this lesson, we'll learn about a two kinds of special layers, not containing any neurons themselves, but that add some functionality that can sometimes benefit a model in various ways. Both are commonly used in modern architectures.

## Dropout Layer

In the last lesson we talked about how overfitting is caused by the network learning spurious patterns in the training data. To recognize these spurious patterns a network will often rely on very a specific combinations of weight, a kind of "conspiracy" of weights. Being so specific, they tend to be fragile: remove one and the conspiracy falls apart. we randomly *drop out* some fraction of a layer's input units every step of training, making it much harder for the network to learn those spurious patterns in the training data. Instead, it has to search for broad, general patterns, whose weight patterns tend to be more robust.

```python
keras.Sequential([
    # ...
    layer.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layer.Dense(16),
    # ...
])
```

## Batch Normalization

+ which can help correct training that is slow or unstable. perhaps with something like scikit-learn's [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) or [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

```python
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
  #--------------------------------------------
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])
```



# Binary Classification.

 neural networks

## 两个精度工具

**Accuracy** is one of the many metrics in use for measuring success on a classification problem

For classification, what we want instead is a distance between *probabilities*, and this is what cross-entropy provides. **Cross-entropy** is a sort of measure for the distance from one probability distribution to another.

**sigmoid activation**.  from 0 to 1.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[33]),
    layers.Dense(4, activation='relu'),    
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

```

```python
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max())) 
```



# #-------------------------

=

---

# Natural Language Processing:

```python
python -m venv tf24
source .env/bin/activate
pip install -U pip setuptools wheel
pip install spacy
#--------------------------------------------
python
import sys
sys.path
sys.path.append('/Users/fyenne/.env/lib/python3.7/site-packages')
```

```python
import spacy
nlp = spacy.load('en')
doc = nlp("Tea is healthy and calming, don't you think?")
for token in doc:
    print(token)
    #--------------------------------------------
# out put all element.
```

## pattern

```python
import spacy

#%%
nlp = spacy.load("en_core_web_sm")
#%%
doc = nlp("Tea is healthy and calming, don't you think?")
#%%
for token in doc:
    print(token)
#%%
print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
    
#--------------------------------------------
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)
#--------------------------------------------
text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.")
matches = matcher(text_doc)
print(matches)
#--------------------------------------------
match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])
```

## 1st project

```python
# data = pd.read_json('../input/nlp-course/restaurant.json')
menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli", ..... ]
#--------------------------------------------
# empty list, store ratings later.
from collections import defaultdict
# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,
# the key is added with an empty list as the value.
item_ratings = defaultdict(list)
# set a matcher up.
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
#--------------------------------------------
for idx, review in data.iterrows(): # slice the dataframe
  # read the text in to nlp ==>
    doc = nlp(review.text)
    # Using the matcher from the previous exercise
    matches = matcher(doc)
    # Create a set of the items found in the review text
    found_items = set([doc[match[1]:match[2]].lower_ for match in matches])
    # Update item_ratings with rating for each item in found_items
    # Transform the item strings to lowercase to make it case insensitive
    for item in found_items:
        item_ratings[item].append(review.stars)
#--------------------------------------------
mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}

# Find the worst item, and write it as a string in worst_item. This can be multiple lines of code if you want.
worst_item = sorted(mean_ratings, key=mean_ratings.get)[0]
#--------------------------------------------

counts = {item: len(ratings) for item, ratings in item_ratings.items()}
item_counts = sorted(counts, key=counts.get, reverse=True)
for item in item_counts:
    print(f"{item:>25}{counts[item]:>5}")
#--------------------------------------------    
sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)

print("Worst rated menu items:")
for item in sorted_ratings[:10]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
    
print("\n\nBest rated menu items:")
for item in sorted_ratings[-10:]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
```











#--------------------------------------------

---

# LDA model:

![Screen Shot 2021-02-09 at 4.22.01 PM](/Users/fyenne/Downloads/booooks/semester5/pythonAAE/py_handbook/pic_for_md/Screen Shot 2021-02-09 at 4.22.01 PM.png)

` Why LDA?`

![Screen Shot 2021-02-09 at 4.26.18 PM](/Users/fyenne/Downloads/booooks/semester5/pythonAAE/py_handbook/pic_for_md/Screen Shot 2021-02-09 at 4.26.18 PM.png)





