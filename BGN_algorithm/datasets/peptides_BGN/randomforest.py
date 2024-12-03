import numpy as np
import pandas as pd
from pep_data_loader import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn import tree
import graphviz
import os
import matplotlib.pyplot as plt

def compute_decision_trees(X, y, column_names):
    regressor = DecisionTreeRegressor(random_state=1,criterion='squared_error')
    parameters = {'max_depth':[2,3,4],
                  'min_samples_split':[2,3,4,5,6],
                  'min_samples_leaf':[2,3,4,5,6]}
    clf = GridSearchCV(regressor, parameters)
    clf.fit(X, y)
    print(clf.best_params_)
    best = DecisionTreeRegressor(random_state=1,
                                 criterion='squared_error',
                                 max_depth=clf.best_params_['max_depth'],
                                 min_samples_split=clf.best_params_['min_samples_split'],
                                 min_samples_leaf=clf.best_params_['min_samples_leaf'])
    best.fit(X, y)
    #best_predict = best.predict(X)
    #print(mean_squared_error(best_predict, y, squared=False))
    #print(r2_score(best_predict, y))
    #print(sum(best.feature_importances_.T > 0))
    #print(best.get_n_leaves())
    dot_data = tree.export_graphviz(best, out_file=None,
                                    feature_names=column_names,
                                    filled=True, rounded=True,
                                    special_characters=True, rotate=True, leaves_parallel=True)
    graph = graphviz.Source(dot_data)
    graph.dpi = 500
    graph.size = "30,30!"
    graph.render(f"decision_tree")
    os.remove(f"decision_tree")
    print("Done!")


def compute_random_forests(X, y):
    regressor = RandomForestRegressor(random_state=1)
    parameters = {'n_estimators':[1, 10, 25, 50, 100],
                  'max_features':["sqrt", "log2", None],
                  'max_depth':[2,3,4,5,6,7],
                  'min_samples_split':[2,3,4,5,6],
                  'min_samples_leaf':[2,3,4,5,6]}
    clf = GridSearchCV(regressor, parameters)
    clf.fit(X, y)
    clf1 = RandomForestRegressor(random_state=1,
                                 criterion='squared_error',
                                 n_estimators=clf.best_params_['n_estimators'],
                                 max_features=clf.best_params_['max_features'],
                                 max_depth=clf.best_params_['max_depth'],
                                 min_samples_split=clf.best_params_['min_samples_split'],
                                 min_samples_leaf=clf.best_params_['min_samples_leaf'])
    clf1.fit(X, y)

    print("Done!")
def compute_feature_importance(X, y):
    regressor = RandomForestRegressor(random_state=1, n_estimators=500)
    regressor.fit(X,y)
    print(pd.DataFrame(regressor.feature_importances_.T, index=X.columns).sort_values(by=0, ascending=False))

def compute_correlation(X):
    cov = np.corrcoef(X.T)
    plt.imshow(cov)
    plt.show()

if __name__ == "__main__":
    X, y = pep_load_data_mean(True, dir="Donnees_peptides_membranes.csv")
    ind = np.arange(len(X))
    ind = shuffle(ind[58:], random_state=10)
    columns_names = X.columns
    X, y = X.to_numpy()[ind], y.to_numpy()[ind]
    compute_decision_trees(X, y, columns_names)

    X, y = pep_load_data_mean(False, dir="Donnees_peptides_membranes.csv")
    ind = np.arange(len(X))
    ind = shuffle(ind[58:], random_state=10)
    columns_names = X.columns
    X, y = X.to_numpy()[ind], y.to_numpy()[ind]
    compute_decision_trees(X, y, columns_names)