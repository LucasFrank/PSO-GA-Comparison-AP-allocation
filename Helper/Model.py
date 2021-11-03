from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
import random
import numpy as npy

class Model():

    def __init__(self, X, Y, model_type = 'MLP', n_neurons = 100, learning_rate = 0.0001, maxiter = 1000, max_depth = None, min_samples_split = 2):
        self.X = X;
        self.Y = Y;
        if(model_type == 'MLP'):
            self.model = self.initialize_mlp(n_neurons, learning_rate, maxiter)
        elif(model_type == 'DT'):
            self.model = self.initialize_dt(max_depth, min_samples_split)

    def initialize_mlp(self, n_neurons, learning_rate, maxiter):
        self.model = MLPRegressor(solver = 'adam', hidden_layer_sizes=(n_neurons),
                                  max_iter = maxiter, early_stopping = True, learning_rate_init = learning_rate)

        return self.model

    def initialize_dt(self, max_depth, min_samples_split):
        self.model = DecisionTreeRegressor(max_features = None, max_depth = max_depth,
                                            min_samples_split = min_samples_split,  splitter = 'best')# criterion = "squared_error",

        return self.model

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate_R2(self, y_test, predicted):
        return r2_score(y_test, predicted)

    def fit_predict_evaluate(self, n_splits, n_repeats):
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
        self.r2score = []

        for train_index, test_index in kf.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            self.fit(x_train, y_train)
            predicted = self.predict(x_test)
            self.r2score.append(self.evaluate_R2(y_test, predicted))

        return npy.mean(self.r2score)
