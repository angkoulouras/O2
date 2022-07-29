import numpy as np
import pandas as pd
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

def train_tabnet(X_train, y_train, eps):
    
    model = TabNetMultiTaskClassifier(n_steps=4) 
    model.fit(X_train, y_train, max_epochs=eps, patience=10)
    
    return model

def train_tabnet2(X_train, y_train, eps):
    
    model = TabNetClassifier(n_steps=4) 
    model.fit(X_train, y_train, max_epochs=eps, patience=10)
    
    return model