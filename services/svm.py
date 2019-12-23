#-*-coding:utf-8-*-

"""
@Class: 人工智能基础与实践
@FileName: 决策树体验
@Author: 上海交通大学
@LastUpdate: 2018.7.13
"""

import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from services.prepare import extract_feature


class SVM():
    def __init__(self,):
        self.model_path = "./model/SVC.model"
        self.train_data = './feature/train/train_data.npy'
        self.train_label = './feature/train/train_label.npy'
        self.val_data = './feature/val/val_data.npy'
        self.val_label = './feature/val/val_label.npy'

        # path relative to engine.py
        self.model_path = "./services/model/SVC.model"

        self.svc = LinearSVC()
        self.svc = joblib.load(self.model_path)

    def train(self, data):
        svc = LinearSVC()

        #x_train = np.load(self.train_data)
        #y_train = np.load(self.train_label)
        #x_valid = np.load(self.val_data)
        #y_valid = np.load(self.val_label)
        
        x_train, y_train, x_valid, y_valid = data 

        svc.fit(x_train, y_train)
        
        accuracy = round(svc.score(x_valid, y_valid))
        
        #print('Test accuracy of SVC: ', accuracy)
        joblib.dump(svc, self.model_path)
        return accuracy

    def predict(self, image,feature_type):
        self.feature_type=feature_type
        X = []
        img = image
        X.append(img)
        X = extract_feature(X, self.feature_type)
        ID = self.svc.predict(X)
        ID_num = ID[0]

        #print('Sign prediction class ID: ', ID_num)
        return ID_num
