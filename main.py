from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_utils import DatasetFactory, fix_columns
from model_tree import ModelTree
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pickle

samples, labels = DatasetFactory('machine')
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.33, random_state=0)

# tree = ModelTree(X_train, y_train, min_samples_leaf=5)
# tree.root = tree.build()
# with open('machine_tree.pkl', 'wb') as output:
#     pickle.dump(tree, output)

with open('machine_tree.pkl', 'rb') as input:
    tree = pickle.load(input)
    preds = [tree.predict(row.to_frame().T) for index, row in X_test.iterrows()]
    print("MODEL TREE:", mean_squared_error(preds, y_test))

    regressor = DecisionTreeRegressor(random_state=0, min_samples_leaf=5)
    train_dummed = pd.get_dummies(X_train)
    regressor.fit(train_dummed, y_train)
    sklearn_preds = regressor.predict(fix_columns(pd.get_dummies(X_test), pd.get_dummies(X_train).columns))
    print("SKLEARN:", mean_squared_error(sklearn_preds, y_test))
