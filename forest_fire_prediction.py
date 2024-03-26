import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import warnings

# import scikit-learn packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix

# Suppress warnings
warnings.filterwarnings("ignore")

# Set display options for pandas
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)

# Loading data
data = pd.read_csv('forestdata.csv')
data.head()

# data cleaning
def clean(rdata):
    """
    Function to clean the dataset.
    Args:
    - rdata: Input DataFrame to be cleaned.

    Returns:
    - nwdata: Cleaned DataFrame.
    """
    # change dots(.) to underscores(_) in column names
    rdata.columns = rdata.columns.str.replace('.', '_')

    # Change day of the day 
    rdata.loc[rdata['time_of_day'] == 'morni7ng', 'time_of_day'] = 'morning'

    # encode time of day and month as onehot approach function
    for cat in rdata['time_of_day'].unique():
        rdata[cat] = np.where(rdata['time_of_day'] == cat, 1, 0)

    for mon in rdata['month'].unique():
        rdata[f'month_{mon}'] = np.where(rdata['month'] == mon, 1, 0)

    # drop missing values and unrequired columns
    rdata = rdata.drop(['collector_id', 'time_of_day', 'month'], axis=1).dropna()

    # trim df
    keep = (rdata['tree_age'] < 250) & (rdata['c_score'] < 250) & (rdata['humidity'] > 30) 
    nwdata = rdata[keep]

    return nwdata

df = clean(data)
df.head()
df.info()

# Pre-modeling

# Select numeric values and categorical values
categ = df.select_dtypes('int')
numer = df.select_dtypes(['float'])

# Standardization of numeric values
scaler = StandardScaler()
for col in numer:
    df[col] = scaler.fit_transform(numer[col].values.reshape(-1, 1))

# Performance Evaluation
def perf_eval(stdmodel, best_model, X, y):
    """
    Function to evaluate model performance.
    Args:
    - stdmodel: Standard model.
    - best_model: Model with optimized hyperparameters.
    - X: Features.
    - y: Target variable.

    Returns:
    - Performance metrics for both standard and hyperparameter tuned models.
    """
    print ('Evaluating the model training data \n')

    def standard():
        pred_y = stdmodel.predict(X)
        accuracy = accuracy_score(y, pred_y)
        return f"Standard Model: accuracy score: {accuracy}"

    def hyper_tuned():
        pred_y = best_model.predict(X)
        accuracy = accuracy_score(y, pred_y)
        return f"Hyper_tuned Model: accuracy score: {accuracy}"

    return standard(), hyper_tuned()

# Confusion Matrix plot
def plot_cm(model, X, y):
    """
    Function to plot confusion matrix.
    Args:
    - model: Trained model.
    - X: Features.
    - y: Target variable.

    Returns:
    - Confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    graph_plot = plot_confusion_matrix(
        model, X, y, labels=best_log_mod.classes_, 
        display_labels=['fire', 'no fire'], ax=ax # check the order
        )
    return graph_plot

# Splitting data into train and test sets
y = df.fire
X = df.drop('fire', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building

# 1. Logistic Regression
logit = LogisticRegression()
log_model = logit.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 2500, 3500, 5000]
}

grid_search = GridSearchCV(logit, param_grid=param_grid, scoring='accuracy', cv=5, verbose=True)

best_log_mod = grid_search.fit(X_train, y_train)

best_log_mod.best_estimator_

# Performance evaluation
# Training data
perf_eval(logit, best_log_mod, X_train, y_train)

# Testing data
perf_eval(logit, best_log_mod, X_test, y_test)

# Confusion Matrix
plot_cm(logit, X_test, y_test)
plot_cm(best_log_mod, X_test, y_test)

# 2. Decision Tree
tree = DecisionTreeClassifier()
tree_model = tree.fit(X_train, y_train)

# Hyperparameter Tuning
param_grid = {
    'max_depth': [3, 5, None],
    'min_samples_leaf': [np.random.randint(1, 10)],
    'criterion': ['gini', 'entropy', 'log_loss']
}

rand_search = GridSearchCV(tree, param_grid, scoring='accuracy', cv=5, verbose=True)

best_tree_mod = rand_search.fit(X_train, y_train)

best_tree_mod.best_estimator_

# Performance evaluation
# Training data
print ('Evaluating the model trained data \n')

# Standard
tree_pred_train_label = tree.predict(X_train)
accuracy = accuracy_score(y_train, tree_pred_train_label)

print(f"Evaluating the Standard Model: accuracy score: {accuracy} \n")

# hyperparameters
tree_pred_train_label = best_tree_mod.predict(X_train)
accuracy = accuracy_score(y_train, tree_pred_train_label)

print(f"Evaluating the Hyperparameter Tuned model: accuracy score: {accuracy}")

# Testing data
print ('Evaluating the model with test data \n')

# Standard
tree_pred_test_label = tree.predict(X_test)
accuracy = accuracy_score(y_test, tree_pred_test_label)

print(f"Evaluating the Standard model: accuracy score: {accuracy} \n")

# hyperparameters
tree_pred_test_label = best_tree_mod.predict(X_test)
accuracy = accuracy_score(y_test, tree_pred_test_label)

print (f"Evaluating the Hyperparameter Tuned model: accuracy score: {accuracy} \n")

# Confusion Matrix
plot_cm(tree, X_test, y_test)
plot_cm(best_tree_mod, X_test, y_test)

# 3. Neural Network
# Standardize the data using a neural network
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

MLP = MLPClassifier()
MLP.fit(X_train, y_train)

# Hyperparameter Tuning
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 100), (10, 50, 10), (20,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'learning_rate_init': [0.001, 0.05],
    'solver': ['sgd', 'adam'],
}

grid_search = GridSearchCV(MLP, param_grid, cv=5, verbose=True, n_jobs=-1)

best_mlp_mod = grid_search.fit(X_train, y_train)

best_mlp_mod.best_estimator_

# Performance evaluation
# Training data
print ('Evaluating the model trained data \n')

# Standard
mlp_pred_train_label = MLP.predict(X_train)
accuracy = accuracy_score(y_train, mlp_pred_train_label)
mse = mean_squared_error(y_train, mlp_pred_train_label)
rmse = np.sqrt(mse)
print(f"Evaluating the Standard Model: accuracy score: {accuracy} \n")

# hyperparameters
mlp_pred_train_label = best_mlp_mod.predict(X_train)
accuracy = accuracy_score(y_train, mlp_pred_train_label)
mse = mean_squared_error(y_train, mlp_pred_train_label)
rmse = np.sqrt(mse)
print(f"Evaluating the Hyperparameter Tuned model: accuracy score: {accuracy}")

# Testing data
print ('Evaluating the model with test data \n')

# Standard
mlp_pred_test_label = MLP.predict(X_test)
accuracy = accuracy_score(y_test, mlp_pred_test_label)
mse = mean_squared_error(y_test, mlp_pred_test_label)
rmse = np.sqrt(mse)
print(f"Evaluating the Standard model: accuracy score: {accuracy} \n")

# hyperparameters
mlp_pred_test_label = best_mlp_mod.predict(X_test)
accuracy = accuracy_score(y_test, mlp_pred_test_label)
mse = mean_squared_error(y_test, mlp_pred_test_label)
rmse = np.sqrt(mse)
print(f"Evaluating the Hyperparameter Tuned model: accuracy score: {accuracy} \n")

# Confusion Matrix
plot_cm(MLP, X_test, y_test)
plot_cm(best_mlp_mod, X_test, y_test)
