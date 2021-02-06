<center><span style="font-size:42px;color:#8B30BB;">Machine Learning Review:  </center></span>

`Siming Yan`

`01,2021`

`According to books: Kaggle, ISLR, Matt:Bussiness data science, AAE722 machine learning in R  `

---

# Kaggle: **Decision tree**

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

Many machine learning models allow some randomness in model training. Specifying a number for `random_state` ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose.

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



 ![Screen Shot 2021-02-04 at 5.30.55 PM](/Users/fyenne/Downloads/booooks/semester5/pythonAAE/py_handbook/Screen Shot 2021-02-04 at 5.30.55 PM.png)



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



# To deal with missing values.

## check NAs.

```python
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
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



### Approach 2 (Imputation)

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



### Approach 3 (An Extension to Imputation)

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

## categorical variables.

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



# cross validation

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

