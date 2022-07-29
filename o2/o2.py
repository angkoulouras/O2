import numpy as np
import pandas as pd
import seaborn as sns
# import xgboost as xgb
import gurobipy as gp
from gurobipy import GRB
import random
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import _tree
from .tabnet_m import train_tabnet, train_tabnet2
from .points_m import generate_points_lr, generate_points_lr_loras, generate_points_lr_adam
from .points_m import generate_points_svm, generate_points_svm_loras 
from .points_m import generate_points_tree, generate_points_tree_loras 
import warnings
import re 
# import tensorflow as tf 

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def find_min_class(X_train, y_train, min_class):
    min_idxs = []
    n, p = X_train.shape
    for i in range(n):
        if(y_train[i]==min_class):
            min_idxs.append(i)
            
    return min_idxs


#removes categorical features from dataframe
#out: a copy of X_train without cat features
def split_cat(X_train):

    n, p = X_train.shape

    ind_num = list() 
    ind_cat = list() 
    for i in range(p): 
        if len(pd.unique(X_train[:,i])) <= 2: 
            ind_cat.append(i)
        else:
            ind_num.append(i)

    return X_train[:, ind_num], X_train[:, ind_cat]


def cat_indeces(X_train):

    n, p = X_train.shape

    ind_num = list() 
    ind_cat = list() 
    for i in range(p): 
        if len(pd.unique(X_train[:,i])) <= 2: 
            ind_cat.append(i)
        else:
            ind_num.append(i)

    return ind_cat

def classify(X_train_new, y_train_new, min_class, clf=None):
    
    if clf is not None:
        y_preds = clf.predict(np.asarray(X_train_new))
        idxs = np.where(y_preds==min_class)
        idxs = list(idxs[0])
        X_train_new = X_train_new[idxs]
        y_train_new = y_train_new[idxs]
    
    return X_train_new, y_train_new

def train_clf(X_train, y_train):
    param_grid_xgb = {
            'min_child_weight': [1],
            'gamma': [0.7],
            'subsample': [0.4, 0.6],
            'colsample_bytree': [0.8],
            'max_depth': [3, 6], 
            'n_estimators': [50, 150, 250],
            'learning_rate': [0.10]
    }

    clf_xgb = xgb.XGBClassifier(objective='binary:logistic')

    grid_xgb = GridSearchCV(estimator = clf_xgb, param_grid = param_grid_xgb, 
                              cv = 3, n_jobs = -1, verbose = 2)
    grid_xgb.fit(X_train, y_train)
    best_xgb = grid_xgb.best_estimator_
    return best_xgb 


def oversample_lr(X_train, y_train, min_class, n_points, l1, l2, l3, optimizer, init_loras):
    (n,p) = X_train.shape 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        min_idxs = find_min_class(X_train, y_train, min_class) 
        if optimizer == "lbfgs":
            if init_loras: 
                X_new, y_new = generate_points_lr_loras(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs) 
            else: 
                X_new, y_new = generate_points_lr(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs)
          
        elif optimizer == "adam":
            X_new, y_new = generate_points_lr_adam(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs)


    # Check for unbounded solutions
    ind_keep = [i for i in range(n,n+n_points) if np.linalg.norm(X_new[i,:]) <= 100]
    X_train_new = X_new[ind_keep]
    y_train_new = y_new[ind_keep]
    y_train_new = min_class*np.ones((len(ind_keep), 1)) 

    return X_train_new, y_train_new


def oversample_svm(X_train, y_train, min_class, n_points, l1, l2, l3, optimizer, init_loras):
    (n,p) = X_train.shape 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        min_idxs = find_min_class(X_train, y_train, min_class) 
        if optimizer == "lbfgs":
            if init_loras: 
                X_new, y_new = generate_points_svm_loras(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs) 
            else: 
                X_new, y_new = generate_points_svm(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs)
          
        elif optimizer == "adam":
            X_new, y_new = generate_points_lr_adam(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs)


    # Check for unbounded solutions
    ind_keep = [i for i in range(n,n+n_points) if np.linalg.norm(X_new[i,:]) <= 100]
    X_train_new = X_new[ind_keep]
    y_train_new = y_new[ind_keep]
    y_train_new = min_class*np.ones((len(ind_keep), 1)) 

    return X_train_new, y_train_new

node_path = "contains node path"

def tree_to_code(tree, feature_names):
    
    global node_path
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth, explored, path_so_far, target_node):
        
        global node_path 

        explored.append(node)
        if (node==target_node):
#             print(path_so_far+" "+str(node))
            node_path = path_so_far+" "+str(node)
            return path_so_far

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if (tree_.children_left[node] not in explored):
                recurse(tree_.children_left[node], depth + 1, explored, path_so_far+ " " +str(node), target_node)
                
            if (tree_.children_right[node] not in explored):
                recurse(tree_.children_right[node], depth + 1, explored, path_so_far+ " " + str(node), target_node)
            

    recurse(0, 1, [], "", np.argmax([tree.tree_.value[i][0][1]/sum(tree.tree_.value[i][0]) for i in range(len(tree.tree_.feature))]))
    
    # get nodes visited in order
    
    nodes = re.split(" ", node_path)
    nodes = [int(nodes[i]) for i in range(1, len(nodes))]
    
    #find constraints
    
    s = ()
    
    for i in range(len(nodes)-1):
        if nodes[i+1]==tree.tree_.children_left[nodes[i]]:
#             print(str(clf.tree_.feature[nodes[i]])+"<="+str(clf.tree_.threshold[nodes[i]]))
            s += ({'type': 'ineq', 'fun': lambda x:  tree.tree_.threshold[nodes[i]] - x[tree.tree_.feature[nodes[i]]]},)
        else:
#             print(str(clf.tree_.feature[nodes[i]])+">"+str(clf.tree_.threshold[nodes[i]]))
            s += ({'type': 'ineq', 'fun': lambda x:  -tree.tree_.threshold[nodes[i]] + x[tree.tree_.feature[nodes[i]]]},)
            
    return s 

def oversample_tree(X_train, y_train, min_class, n_points, l1, l2, l3, optimizer, init_loras):
    (n,p) = X_train.shape 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        min_idxs = find_min_class(X_train, y_train, min_class) 
        clf = DecisionTreeClassifier(random_state=0) 
        clf.fit(X_train, y_train)  
        df = pd.DataFrame(X_train)  
        features = [col for col in df.columns] 
        cons = tree_to_code(clf, [str(i) for i in range(clf.n_features_)]) 

        if optimizer == "lbfgs":
            if init_loras: 
                X_new, y_new = generate_points_tree_loras(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs, cons)
            else: 
                X_new, y_new = generate_points_tree(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs, cons)
          
        elif optimizer == "adam":
            X_new, y_new = generate_points_lr_adam(X_train, y_train, min_class, n_points, l1, l2, l3, min_idxs)


    # Check for unbounded solutions
    ind_keep = [i for i in range(n,n+n_points) if np.linalg.norm(X_new[i,:]) <= 100]
    X_train_new = X_new[ind_keep]
    y_train_new = y_new[ind_keep]
    y_train_new = min_class*np.ones((len(ind_keep), 1)) 

    return X_train_new, y_train_new



def oversample_random(X_train, y_train, min_class, n_points):
    new_p = 0
    n, p = X_train.shape
    X_train_new = np.zeros((n_points, p))
    y_train_new = min_class*np.ones((n_points, 1))
    x_new = np.zeros((1, p))
    
    #indices of min_class samples
    min_idxs = find_min_class(X_train, y_train, min_class)
    
    #select n_points from minory class
    min_idxs = np.array(min_idxs)[np.random.choice(len(min_idxs), size=n_points, replace=False).astype(int)]
    
    for i in min_idxs:
        
        #add noise to point from min_class
        x_new = X_train[i, :] + np.random.randn(1, p)/3
        
        #add new point
        X_train_new[new_p] = x_new
        new_p += 1
        
    return X_train_new, y_train_new

       
def oversample_knn(X_train, y_train, min_class, nn, n_points):

    new_p = 0
    n, p = X_train.shape
    X_train_new = np.zeros((n_points, p))
    y_train_new = min_class*np.ones((n_points, 1))
    x_new = np.zeros((1, p))

    #find nn of each point
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)

    #indices of min_class samples
    min_idxs = find_min_class(X_train, y_train, min_class)
    
    #select n_points from minory class
    min_idxs = np.array(min_idxs)[np.random.choice(len(min_idxs), size=n_points, replace=False).astype(int)]
       
    for i in min_idxs:
        nn_idxs = indices[i]

        #model
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            m = gp.Model('Gurobi', env=env)

        #variables
        new_x = m.addVars(p, lb=-float('inf'))

        #objective
        sum1 = 0
        for idx in nn_idxs:
            if idx!=i:
                for j in range(p):
                    sum1 += (X_train[idx,j]-new_x[j])*(X_train[idx,j]-new_x[j])
        m.setObjective(sum1, GRB.MINIMIZE)

        #optimize
        m.optimize()

        #add new point
        for j in range(p):
            x_new[0, j] = new_x[j].x

        X_train_new[new_p] = x_new
        new_p += 1
        
    return X_train_new, y_train_new

def thold(X_train, ind_cat):

    tholds = np.zeros(len(ind_cat))
    j=0
    for i in ind_cat:
        tholds[j] = np.mean(X_train[:, i], axis=0)
        j += 1
    return tholds

def ovs(X_train, y_train, min_class, points, method="cat", clf=None, ovs_m="lr", eps=[10, 10], l1=0.5, l2=0.5, l3=0.5, optimizer="lbfgs", init_loras=False):

    X_train_ov = X_train
    n_points = points
    
    if(method=="cat_tab"):
        X_train_ov, X_train_cat = split_cat(X_train)
        tabnet_model = train_tabnet(X_train_ov, X_train_cat, eps[0])
        num_cat = X_train_cat.shape[1]
    elif(method=="cat_mean"):
        ind_cat = cat_indeces(X_train)
        thresholds = thold(X_train, ind_cat)
        num_cat = len(ind_cat) 

    #create new samples
    if ovs_m=="lr":
        X_new, y_new = oversample_lr(X_train_ov, y_train, min_class, n_points, l1, l2, l3, optimizer, init_loras) 
    elif ovs_m=="svm":
        X_new, y_new = oversample_svm(X_train_ov, y_train, min_class, n_points, l1, l2, l3, optimizer, init_loras)
    elif ovs_m=="tree":
        X_new, y_new = oversample_tree(X_train_ov, y_train, min_class, n_points, l1, l2, l3, optimizer, init_loras) 

     
    # predict cat features for new samples
    if(method=="cat_tab"):
        X_new_cat = np.array(tabnet_model.predict(X_new)).reshape((X_new.shape[0], num_cat)) 
        X_new = np.hstack([X_new, X_new_cat])
    elif(method=="cat_mean"):
        n, p = X_new.shape
        for i in range(n):
            for j, k in enumerate(ind_cat):
                if X_new[i, k]>= thresholds[j]:
                    X_new[i, k] = 1.0
                else:
                    X_new[i, k] = 0.0
        
    #keep the ones that belong to the min_class
    if clf is not None:
        if clf == "xgb":
            clf = train_clf(X_train, y_train)
        elif clf == "tabnet":
            clf = train_tabnet2(X_train, y_train, eps[1])
            
        X_new = X_new.astype(np.float)
        X_new, y_new = classify(X_new, y_new, min_class, clf)
    
    print(X_new.shape)
    #merge with train set
    X_train_new = np.vstack((X_train, X_new))
    y_train_new = np.hstack((y_train, y_new[:, 0]))
    

    return X_train_new, y_train_new