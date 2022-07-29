import numpy as np 
import pandas as pd 
# import xgboost as xgb 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs 
from scipy.optimize import minimize  
import gurobipy as gp 
from gurobipy import GRB 
import loras 

def get_beta(X,y): 
    (n,p) = X.shape  
    clf = LogisticRegression(random_state=10) 
    clf.fit(X, y) 
    beta = clf.coef_.reshape((p,)) 
    return beta 

def get_hyperplane(X,y): 
    C = 0.1
    (n,p) = X.shape  
    env = gp.Env(empty=True) 
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model(env=env) 
    w = m.addVars(p,  name="Normal") 
    b = m.addVar(1, name="bias") 
    u = m.addVars(n,  name="Auxiliary") 
    m.addConstrs(y[i]*(sum([w[j]*X[i,j] for j in range(p)])+b) >= 1 - u[i] for i in range(n)) 
    m.addConstrs(u[i] >= 0 for i in range(n))
    m.setObjective(sum([w[j]**2 for j in range(p)]) + C*u.sum(), GRB.MINIMIZE) 
    m.optimize() 
    w_final = [w[j].x for j in w.keys()] 
    b_final = b.x
    return w_final, b_final

# get more starting points

def get_more_points(X_train, y_train):
    
    features=X_train 
    labels=y_train 
    label_1=np.where(labels == 1)[0] 
    label_1=list(label_1) 
    features_1=features[label_1] 
    label_0=np.where(labels == 0)[0] 
    label_0=list(label_0) 
    features_0=features[label_0] 
    training_data=np.concatenate((features_1,features_0)) 
    training_labels=np.concatenate((np.zeros(len(features_1))+1, 
                                np.zeros(len(features_0))))
    min_class_points = features_1 
    maj_class_points = features_0
    loras_min_class_points = loras.fit_resample(maj_class_points, 
                                            min_class_points,
                                            num_generated_points=1) 
    
    return loras_min_class_points

# We will use a separate matrix to store the newly generated data points 
def obj_fun(x, beta, prev_points, l1, l2, l3, x0, X_train, min_idxs): 
    k = prev_points.shape[0] 
    pen1 = l1*sum([np.linalg.norm(x - prev_points[i,:])**2 for i in range(k)]) / k
    pen2 = l2 * sum([np.linalg.norm(x-X_train[min_idxs[i], :])**2 for i in range(len(min_idxs))]) / len(min_idxs)
    pen3 = -l3 * np.linalg.norm(x)**2
    return -(np.exp(np.dot(beta,x)) / (1.0 + np.exp(np.dot(beta, x)) ) + pen1 + pen2 + pen3) 

def generate_points_lr(X_train, y_train, min_class, k, l1, l2, l3, min_idxs): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    beta = get_beta(X, y)
    
    if(k>len(min_idxs)):
        new_points = get_more_points(X_train, y_train)
        min_idxs_new = np.array(min_idxs)[np.random.choice(len(min_idxs), size=len(min_idxs), replace=False).astype(int)]
    else:
        min_idxs_new = np.array(min_idxs)[np.random.choice(len(min_idxs), size=k, replace=False).astype(int)]
    
    for i in range(k):
        if (k<len(min_idxs)):
            x0 = X_train[min_idxs_new[i], :]
        else:
            x0 = new_points[k-len(min_idxs)+1, :]
#         x0 = np.ones(p)
        opt = fmin_l_bfgs_b(obj_fun, x0=x0, args=(beta, prev_points, l1, l2, l3, x0, X_train, np.array(min_idxs)), approx_grad=True) 
        sol = opt[0] 
        X = np.vstack([X, sol]) 
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y 

def generate_points_lr_loras(X_train, y_train, min_class, k, l1, l2, l3, min_idxs): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    beta = get_beta(X, y)
    
    features=X_train 
    labels=y_train 
    label_1=np.where(labels == 1)[0] 
    label_1=list(label_1) 
    features_1=features[label_1] 
    label_0=np.where(labels == 0)[0] 
    label_0=list(label_0) 
    features_0=features[label_0] 
    training_data=np.concatenate((features_1,features_0)) 
    training_labels=np.concatenate((np.zeros(len(features_1))+1, np.zeros(len(features_0))))
    min_class_points = features_1 
    maj_class_points = features_0 
    loras_min_class_points = loras.fit_resample(maj_class_points, min_class_points,num_generated_points=3) 
    n1 = min_class_points.shape[0]
    n2 = loras_min_class_points.shape[0] 
    min_idxs_new = np.random.choice(np.arange(n1,n2), size=k, replace=False).astype(int)

    for i in range(k):
        x0 = loras_min_class_points[min_idxs_new[i], :]
        opt = fmin_l_bfgs_b(obj_fun, x0=x0, args=(beta, prev_points, l1, l2, l3, x0, X_train, np.array(min_idxs)), approx_grad=True) 
        sol = opt[0] 
        X = np.vstack([X, sol]) 
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y 

def generate_points_lr_adam(X_train, y_train, min_class, k, l1, l2, l3, min_idxs, steps=100): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    beta = get_beta(X, y) 
    min_idxs = np.array(min_idxs)[np.random.choice(len(min_idxs), size=k, replace=False).astype(int)] 
    for i in range(k):     
        x0 = X_train[min_idxs[i], :] 
        opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
        x = tf.Variable(x0) 
        k1 = prev_points.shape[0] 
        loss = lambda: -1.0 / (1.0 + tf.exp(sum(-beta[i]*x[i] for i in range(p)))) + sum([l1 * tf.norm(x - prev_points[i,:])**2 for i in range(k1)]) 
        - (l3 * tf.norm(x)**2) # + (l2 * sum((x[i] - x0[i])**2 for i in range(p)))
        for i in range(steps): 
            opt.minimize(loss, [x])
        sol = x.numpy()  
        X = np.vstack([X, sol])   
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y


def obj_fun_svm(x, w, b, prev_points, l1, l2, l3, x0, X_train, min_idxs): 
    k = prev_points.shape[0] 
    pen1 = l1*sum([np.linalg.norm(x - prev_points[i,:])**2 for i in range(k)]) / k
    pen2 = l2 * sum([np.linalg.norm(x-X_train[min_idxs[i], :])**2 for i in range(len(min_idxs))]) / len(min_idxs)
    pen3 = -l3 * np.linalg.norm(x)**2
    return -(np.sign(np.dot(w,x) + b) + pen1 + pen2 + pen3) 


def generate_points_svm(X_train, y_train, min_class, k, l1, l2, l3, min_idxs): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    w, b = get_hyperplane(X,y)  
    
    if(k>len(min_idxs)):
        new_points = get_more_points(X_train, y_train)
        min_idxs_new = np.array(min_idxs)[np.random.choice(len(min_idxs), size=len(min_idxs), replace=False).astype(int)]
    else:
        min_idxs_new = np.array(min_idxs)[np.random.choice(len(min_idxs), size=k, replace=False).astype(int)]
    
    for i in range(k):
        if (k<len(min_idxs)):
            x0 = X_train[min_idxs_new[i], :]
        else:
            x0 = new_points[k-len(min_idxs)+1, :]
#         x0 = np.ones(p)
        opt = fmin_l_bfgs_b(obj_fun_svm, x0=x0, args=(w, b, prev_points, l1, l2, l3, x0, X_train, np.array(min_idxs)), approx_grad=True) 
        sol = opt[0] 
        X = np.vstack([X, sol]) 
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y 


def generate_points_svm_loras(X_train, y_train, min_class, k, l1, l2, l3, min_idxs): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    w, b = get_hyperplane(X,y) 
    
    features=X_train 
    labels=y_train 
    label_1=np.where(labels == 1)[0] 
    label_1=list(label_1) 
    features_1=features[label_1] 
    label_0=np.where(labels == 0)[0] 
    label_0=list(label_0) 
    features_0=features[label_0] 
    training_data=np.concatenate((features_1,features_0)) 
    training_labels=np.concatenate((np.zeros(len(features_1))+1, np.zeros(len(features_0))))
    min_class_points = features_1 
    maj_class_points = features_0 
    loras_min_class_points = loras.fit_resample(maj_class_points, min_class_points,num_generated_points=3) 
    n1 = min_class_points.shape[0]
    n2 = loras_min_class_points.shape[0] 
    min_idxs_new = np.random.choice(np.arange(n1,n2), size=k, replace=False).astype(int)

    for i in range(k):
        x0 = loras_min_class_points[min_idxs_new[i], :]
        opt = fmin_l_bfgs_b(obj_fun_svm, x0=x0, args=(w, b, prev_points, l1, l2, l3, x0, X_train, np.array(min_idxs)), approx_grad=True) 
        sol = opt[0] 
        X = np.vstack([X, sol]) 
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y

# We will use a separate matrix to store the newly generated data points 
def obj_fun_tree(x, prev_points, l1, l2, l3, x0, X_train, min_idxs): 
    k = prev_points.shape[0] 
    pen1 = l1*sum([np.linalg.norm(x - prev_points[i,:])**2 for i in range(k)]) / k
    pen2 = l2 * sum([np.linalg.norm(x-X_train[min_idxs[i], :])**2 for i in range(len(min_idxs))]) / len(min_idxs)
    pen3 = -l3 * np.linalg.norm(x)**2
    return -(pen1 + pen2 + pen3) 

def generate_points_tree(X_train, y_train, min_class, k, l1, l2, l3, min_idxs, cons): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    
    if(k>len(min_idxs)):
        new_points = get_more_points(X_train, y_train)
        min_idxs_new = np.array(min_idxs)[np.random.choice(len(min_idxs), size=len(min_idxs), replace=False).astype(int)]
    else:
        min_idxs_new = np.array(min_idxs)[np.random.choice(len(min_idxs), size=k, replace=False).astype(int)]
    
    for i in range(k):
        if (k<len(min_idxs)):
            x0 = X_train[min_idxs_new[i], :]
        else:
            x0 = new_points[k-len(min_idxs)+1, :]
        opt = minimize(obj_fun_tree, x0=x0, method='SLSQP', args=(prev_points, l1, l2, l3, x0, X_train, np.array(min_idxs)), constraints=cons) 
        sol = opt.x 
        X = np.vstack([X, sol]) 
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y 


def generate_points_tree_loras(X_train, y_train, min_class, k, l1, l2, l3, min_idxs, cons): 
    X = X_train 
    y = y_train  
    (n,p) = X.shape 
    prev_points = np.zeros((1,p))
    w, b = get_hyperplane(X,y) 
    
    features=X_train 
    labels=y_train 
    label_1=np.where(labels == 1)[0] 
    label_1=list(label_1) 
    features_1=features[label_1] 
    label_0=np.where(labels == 0)[0] 
    label_0=list(label_0) 
    features_0=features[label_0] 
    training_data=np.concatenate((features_1,features_0)) 
    training_labels=np.concatenate((np.zeros(len(features_1))+1, np.zeros(len(features_0))))
    min_class_points = features_1 
    maj_class_points = features_0 
    loras_min_class_points = loras.fit_resample(maj_class_points, min_class_points,num_generated_points=3) 
    n1 = min_class_points.shape[0]
    n2 = loras_min_class_points.shape[0] 
    min_idxs_new = np.random.choice(np.arange(n1,n2), size=k, replace=False).astype(int)

    for i in range(k):
        x0 = loras_min_class_points[min_idxs_new[i], :]
        opt = minimize(obj_fun_tree, x0=x0, method='SLSQP', args=(prev_points, l1, l2, l3, x0, X_train, np.array(min_idxs)), constraints=cons) 
        sol = opt.x 
        X = np.vstack([X, sol]) 
        prev_points = np.vstack([prev_points, sol]) 
        y = np.append(y, min_class) 
    return X,y
 