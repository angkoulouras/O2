

## Usage 

Generate points by calling the function ovs(X_train, y_train, min_class, points, method=method, clf=None, 
                                            ovs_m=ovs_m, eps=[10, 10], l1=0.5, l2=0.5, l3=0.5, optimizer="lbfgs", init_loras=False)

### Mandatory Inputs 

1) X_train, y_train: the training data and labels.
2) min_class: the minority class label. 
3) points: the number of new data points to be generated.

### Optional Inputs

1) clf: Whether to use classifier for filtering or not. Takes values \{ None, xgb, tabnet\}. 
2) ovs_m: Which model to use as Model A. Takes values \{ "lr", "svm", tabnet\}. 



### Output 

X_train_new, y_train_new: Training data and labels concatenated with synthetic points and min class labels respectively.