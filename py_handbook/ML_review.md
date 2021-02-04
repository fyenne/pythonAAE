<center><span style="font-size:42px;color:#8B30BB;">Machine Learning Review:  </center></span>

`siming Yan`

`01,2021`

`According to books: Kaggle, ISLR, Matt:Bussiness data science, AAE722 machine learning in R  `

---

## 1. Kaggle: Decision tree

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

---

### 1.1 Traning model



![Screen Shot 2021-02-04 at 5.30.55 PM](/Users/fyenne/Downloads/booooks/semester5/tensor/Screen Shot 2021-02-04 at 5.30.55 PM.png)



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

### 1.2 Fill'in best parameters and re-train the model.

```python
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size,random_state = 1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)
```

---

## 2. Kaggle Random Forrests. Scikit Learning.



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

### 2.1 write a random forrest Parameter choosing function:

```python
# n_estimators, max_depth 
# choose two parameters.:
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

