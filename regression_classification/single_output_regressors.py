# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:14:42 2018

@author: spele
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
#from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import  mutual_info_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from itertools import product
#from multiprocessing import Pool
#from functools import partial



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dir')
    argparser.add_argument('-s', '--shifted', type=int, default=0)
#    argparser.add_argument('-iter', type=int, default=5)
#    argparser.add_argument('-f', '--file', action='append')
    parameters = vars(argparser.parse_args())
#    print(parameters['file'][0], parameters['file'][1])
    #==============================================================================
    # IMPORT FINAL2
    #==============================================================================
    directory = "datasets/"+parameters['dir']+"/"
    print("dataset:",directory)    
    x = pd.read_csv(directory+"x_"+parameters['dir'][:-3]+"_2017.csv")
    y = pd.read_csv(directory+"y_"+parameters['dir'][:-3]+"_2017.csv")
    
    
    clusters = ['A', 'B', 'C', 'D', 'E', 'F']
    cond = [a for a in list(x) if a not in clusters+['Time index', 'Total users']]
    
    
    shift_for = parameters["shifted"]
#    iterat = parameters['iter']
    print("shift_for:",shift_for)
#    print("n_iter for RandomizedSearchCV:",iterat)
    x[cond] = x[cond].shift(-shift_for)
    x = x.fillna(method='ffill')
    
    
    #==============================================================================
    # Regression preprocessing
    #==============================================================================
    
    
    
    # dividing X, y into train and test data for one cluster
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 42) 
    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    
    
    
    # =============================================================================
    # FEATURE SELECTION
    # =============================================================================
    
    
    selectK_mutual_info = {el: 0 for el in clusters}
    
    for i in range(len(clusters)):
        mutual = SelectKBest(mutual_info_regression, k=12)
        mutual.fit(X_train, y_train[:,i])
        selectK_mutual_info[clusters[i]] = dict(zip(list(x), mutual.scores_))
    

    metrics = pd.DataFrame(index=list(x), columns = clusters)
    
    for cl in clusters:
        for values in selectK_mutual_info[cl]:
            metrics[cl][values] = selectK_mutual_info[cl][values]
        
# =============================================================================
#PIPELINE-PREVIOUS
# =============================================================================
    
        
#    def loss(y_test, y_pred):
#        return np.sum(np.equal(y_test, y_pred))/float(y_test.size)
#    
#    acc = make_scorer(loss, greater_is_better=True)
#    
#    
#    classifier = OneVsRestClassifier(SVC(random_state=0))
#    
#    pipe_svm = Pipeline([('scl', StandardScaler()), ('class', MultiOutputClassifier(classifier))])
#    
#    
#    param_dist = {"class__estimator__estimator__C": stats.uniform(1e-1, 3),
#                  "class__estimator__estimator__gamma": stats.uniform(1e-2, 100)}
#    
#    
#    clf = RandomizedSearchCV(estimator = pipe_svm,iid =True, cv=5, scoring = acc,
#                             param_distributions= param_dist, n_iter=100)
#    
##    clf.get_params().keys()
##    parameters = clf.get_params()
#    clf.fit(X_train, y_train)
##    clf.best_estimator_
#    
#    y_pred = clf.predict(X_test)
            
# =============================================================================
# ESTIMATORS NEW
# =============================================================================
    def create_fit_estimator(estim, X_train_, y_train_, X_test_, y_test_):
        
        combos =    {"+time": [6,7,9,10],
                     "+time_pops": [9,10],
                     "+time_temp/cond": [6,7],
                     "+time_pops_temp/cond": []
                     }
        
        accuracy = {}

        for combo in combos:
            X_tr = np.delete(X_train_, combos[combo], axis =1)
            X_te = np.delete(X_test_, combos[combo], axis =1)
            
            pipe_svm = Pipeline([('scl', StandardScaler()), ('class', estim["estimator"])])
        
            clf = GridSearchCV(estimator = pipe_svm,iid =True, cv=5, scoring = "neg_mean_squared_error",
                                 param_grid=estim["param_dist"])
            try:
                clf.fit(X_tr, y_train_)
            except ValueError:
                print("error for"+estim["estimator"])
                clf = pipe_svm
                clf.fit(X_tr, y_train_)
     
            y_pred = clf.predict(X_te)
            
            accuracy[combo] = mean_absolute_error(y_test_, y_pred)/np.mean(y_test)
        
        return accuracy
    
    mult_SVR = SVR()
    mult_KR = KernelRidge(kernel='rbf')
    mult_RF = RandomForestRegressor()
    mult_KNN = KNeighborsRegressor()
    
    RF_param_dist = {"class__n_estimators": [int(x) for x in np.linspace(start = 10, stop = 40, num = 10)],
                     "class__max_depth" : [int(x) for x in np.linspace(3, 20, num = 11)],
                     "class__min_samples_leaf": [1, 2, 4]}
    
    KNN_param_dist = {"class__n_neighbors" : range(1, 31),
                      "class__weights" : ['uniform', 'distance']}
    
    SVR_param_dist = {"class__C": np.linspace(1e-2, 3, 20),
                      "class__gamma": np.linspace(1e-2, 100, 20)}
    
    KR_param_dist = {"class__alpha": np.linspace(1e-1, 3, 20),
                     "class__gamma": np.linspace(1e-2, 100, 100)}
    
    
    estimators = {"mult_RF":{"estimator": mult_RF,"param_dist":RF_param_dist} ,
                  "mult_KNN":{"estimator": mult_KNN,"param_dist":KNN_param_dist}, 
                  "mult_SVR":{"estimator": mult_SVR,"param_dist":SVR_param_dist}, 
                  "mult_KR":{"estimator": mult_KR,"param_dist":KR_param_dist}}

#    pool = Pool(4)
#    results = pool.map(partial(create_fit_estimator, X_train_ = X_train, y_train_ = y_train,
#                               X_test_ = X_test, y_test_ = y_test, n_it=iterat), estimators)


    skata = {}
    for keys in estimators:
        skata[keys] = {}
        for i in range(6):
            skata[keys][chr(i+65)] = create_fit_estimator(estimators[keys], X_train_ = X_train, y_train_ = y_train[:,i],
                                   X_test_ = X_test, y_test_ = y_test[:,i])



# =============================================================================
# WRITE METRICS+RESULTS
# =============================================================================

    accuracies = pd.DataFrame(index=list(estimators.keys()), 
                              columns = [x[0]+x[1] for x in list(product(clusters, list(skata[keys]["A"].keys())))])

    for keys in skata:
        for combinations in list(accuracies.columns):
            accuracies[combinations][keys] = skata[keys][combinations[0]][combinations[1:]]

#    print ("ACC:", np.sum(np.equal(y_test, y_pred))/float(y_test.size))
    
    metrics.to_csv(directory+'regression_metrics_'+parameters['dir'][:-3]+'_SHIFT_'+str(shift_for)+'.csv')
    accuracies.to_csv(directory+'regression_accuracies_'+parameters['dir'][:-3]+'_SHIFT_'+str(shift_for)+'.csv')    
    
    
#    f1_score(y_test[:,2], y_pred[:,2], average = "micro")
    
    
    
    
    
    
    
    
    
    
    
    
