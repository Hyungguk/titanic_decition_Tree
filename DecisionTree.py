from collections import Counter
import numpy as np
import pandas as pd
from numpy import genfromtxt
import scipy.io
from scipy import stats
import csv
from csv import DictReader
import sklearn
from sklearn import preprocessing
import random
import math
import matplotlib.pyplot as plt
from save_csv import results_to_csv

class DecisionTree:
    def __init__(self, splitRule=None, depth=0, isLeaf=False, maxDepth=22, isRandom=False, features=[]):
        self.splitRule = splitRule
        self.left = None
        self.right = None
        self.label = None
        self.depth = depth
        self.isLeaf = isLeaf
        self.numNodes = 0
        self.maxDepth = maxDepth
        self.features = features
        self.isRandom = isRandom

    def get_idx(self):
        return self.splitRule[0]

    def get_threshold(self):
        return self.splitRule[1]

    def get_label(self):
        return self.label

    def setMaxDepth(self, i):
        self.maxDepth = i

        
    @staticmethod
    def entropy(y):
        classes = np.unique(y)
        h = 0
        for c in classes:
            prior = y.count(c) / len(y) 
            h -= prior * np.log2(prior)
        return h

    @staticmethod
    def information_gain(X, y, thresh):
        h = DecisionTree.entropy(y)
        groupA = []
        groupB = []
        for i in range(len(y)):
            if X.item(i) < thresh:
                groupA.append(y[i])
            else:
                groupB.append(y[i])
        h_l = DecisionTree.entropy(groupA)
        h_r = DecisionTree.entropy(groupB)
        h_after = (len(groupA)*h_l + len(groupB)*h_r) / len(y)
        return h - h_after

    @staticmethod
    def gini_impurity(y):
        classes = np.unique(y)
        h = 0
        count = Counter(y)
        for c in classes:
            prior = count[c] / len(y)
            h += prior * (1-prior)
        return h

    @staticmethod
    def gini_purification(x, y, thresh):
        g = DecisionTree.gini_impurity(y)
        groupA = []
        groupB = []
        for i in range(len(y)):
            if X.item(i) < thresh:
                groupA.append(y[i])
            else:
                groupB.append(y[i])
        g_l = DecisionTree.gini_impurity(groupA)
        g_r = DecisionTree.gini_impurity(groupB)
        g_after = (len(groupA)*g_l + len(groupB)*g_r) / len(y)
        return g - g_after

    def split(self, X, y, idx, thresh):
        X_l = []
        X_r = []
        y_l = []
        y_r = []

        for i in range(len(X)):
            if X[i][idx] < thresh:
                X_l.append(X[i])
                y_l.append(y[i])
            else:
                X_r.append(X[i])
                y_r.append(y[i]) 
        return X_l, y_l, X_r, y_r
    
    def segmenter(self, X, y):
        best_split = None
        best_gini = -float('inf')
        if self.isRandom:
            r = self.features
        else:
            r = [i for i in range(len(X[0]))]
        for f in r:
            x = np.transpose(X)[f]
            thresh = min(x)+1
            while thresh <= max(x):
                gini = DecisionTree.gini_purification(x, y, thresh)
                if best_gini <= gini:
                    best_split = (f, thresh)
                    best_gini = gini
                if max(x) - min(x) > 10:
                    thresh += (max(x) - min(x)) / 10
                else:
                    thresh += 1
        return best_split
     
    
    def fit(self, X, y):
        l = Counter(y).most_common(1)
        if len(y) == 0:
            return None
        if l[0][1] > len(y)*0.95:
            self.label = l[0][0]
            self.isLeaf = True
        else:
            splitRule = self.segmenter(X, y)
            if splitRule != None:
                idx = splitRule[0]
                thresh = splitRule[1]
                self.splitRule = splitRule
            else:
                self.label = l[0][0]
                self.isLeaf = True
                return

            if self.depth >= self.maxDepth:
                self.label = l[0][0]
                self.isLeaf = True
            else:
                X_l, y_l, X_r, y_r = self.split(X, y, idx, thresh)
                if self.isRandom:
                    m = math.ceil(math.sqrt(len(X[0])))
                    indices = [i for i in range(len(X[0]))]
                    np.random.shuffle(indices)
                    f = indices[:m]
                    self.left = DecisionTree(depth=self.depth+1, maxDepth=self.maxDepth, isRandom=True, features=f)
                    self.right = DecisionTree(depth=self.depth+1, maxDepth=self.maxDepth, isRandom=True, features=f)
                else:
                    self.left = DecisionTree(depth=self.depth+1, maxDepth=self.maxDepth)
                    self.right = DecisionTree(depth=self.depth+1, maxDepth=self.maxDepth)
                self.left.fit(X_l, y_l)
                self.right.fit(X_r, y_r)
    
    def predict(self, X):
        predictions = []
        for x in X:
            curr_node = self
            while not curr_node.isLeaf:
                idx = curr_node.get_idx()
                thresh = curr_node.get_threshold()
                if x[idx] < thresh:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
            predictions.append(curr_node.label)
        return predictions

    @staticmethod
    def errorRate(predictions, actual):
        """
        Evaluate the error rate given predictions and actual labels.
        """
        err_count = 0
        for i in range(len(predictions)):
            if predictions[i] != actual[i]:
                err_count += 1
        return err_count/len(predictions)

    def __repr__(self, depth=0):
        ret = ""
        if self.right != None:
            ret += self.right.__repr__(depth + 1)
        if self.isLeaf:
            ret += "\n" + ("    "*depth) + "label: " + str(self.label)
        elif self.splitRule:
            ret += "\n" + ("    "*depth) + str(self.splitRule)
        if self.left != None:
            ret += self.left.__repr__(depth + 1)

        return ret


class RandomForest():
    
    def __init__(self, numTrees, maxDepth=22):
        self.numTrees = numTrees
        self.trees = []
        for i in range(self.numTrees):
            tree = DecisionTree(maxDepth=maxDepth, isRandom=True)
            self.trees.append(tree)
        

    def fit(self, X, y):
        for tree in self.trees:
            """
            Get random subsamples of training Data (X, y) 
            for each DecitionTree (BAGGING).
            """
            X_sub = []
            y_sub = []
            while len(y_sub) < len(y)*0.632:
                random_i = np.random.randint(0, len(y))
                X_sub.append(X[random_i])
                y_sub.append(y[random_i])
            """
            Generate random indices of features so that each tree
            can train with random subsamples of features. 
            """
            m = math.ceil(math.sqrt(len(X[0])))
            indices = [i for i in range(len(X[0]))]
            np.random.shuffle(indices)
            tree.features = indices[:m]
            """
            Train each tree with its subsampled training data.
            """
            tree.fit(X_sub, y_sub)

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        pred_per_trees = []
        for tree in self.trees:
            pred_per_trees.append(tree.predict(X))
        predictions = []
        for pred in np.transpose(pred_per_trees):
            l = Counter(pred).most_common(1)[0][0]
            predictions.append(l)
        return predictions
    
if __name__ == "__main__":
    #dataset = "titanic"
    dataset = "spam"

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv' 
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        
        # Notes: 
        # 1. Some data points are missing their labels
        # 2. Some features are not numerical but categorical
        # 3. Some values are missing for some features
        """
        Handle missing values and categorical features.
        """
        data_new = []
        age = []
        fare = []
        embarked = []
        """
        Get the median age and fare of died and survived to replace
        missing values. 
        For embarked, get the most common value for each label.
        """
        for x in data[1:]:
            if x[3].decode('utf8') != '':
                age.append(float(x[3].decode('utf8')))
            if x[7].decode('utf8') != '':
                fare.append(float(x[7].decode('utf8'))) 
            if x[9].decode('utf8') != '':
                embarked.append(x[9].decode('utf8'))
        med_age = np.median(age)
        med_fare = np.median(fare)
        maj_embarked = Counter(embarked).most_common(1)[0][0]

        for x in data[1:]:
            if x[0].decode('utf8') != None:
                """
                Do not use training_data with unknown label.
                """
                new_x = []
                for i in range(len(x)):
                    x_i = x[i].decode('utf8')
                    if i == 1:
                        """
                        Feature pclass.
                        Handle the categorical feature by vectorizing it.
                        """
                        if x_i == '1':
                            new_x = new_x + [1, 0, 0]
                        elif x_i == '2':
                            new_x = new_x + [0, 1, 0]
                        elif x_i == '3':
                            new_x = new_x + [0, 0, 1]
                    elif i == 3:
                        """
                        Feature age.
                        Handle the missing features by replacing the None
                        values with the medican among ones in the same class.
                        """
                        if not x_i:
                            new_x.append(med_age)
                        else:
                            new_x.append(float(x_i))
                    elif i == 2:
                        """
                        Feature sex.
                        Map to 0 if male, 1 if Female.
                        """
                        if x_i == 'male':
                            new_x.append(0)
                        elif x_i == 'female':
                            new_x.append(1)

                    elif i == 9:
                        """
                        Feature embarked.
                        Handle the categorical feature by vectorizing it.
                        """
                        if x_i == '':
                            x_i = maj_embarked
                        if x_i == 'C':
                            new_x = new_x + [1, 0, 0]
                        elif x_i == 'Q':
                            new_x = new_x + [0, 1, 0]
                        elif x_i == 'S':
                            new_x = new_x + [0, 0, 1]
                    elif i == 7:
                        """
                        Feature fare.
                        Handle missing values by replacing it with median.
                        """
                        if x_i == '':
                            new_x.append(med_fare)
                        else:
                            new_x.append(float(x_i))
                    elif i != 0 and i != 6 and i != 8:
                        """
                        Not use 'cabin' feature because there are too many missing values.
                        Not use 'ticket' features because the formats are all so different and
                        the ticket number seems to be independent with survival.
                        For other cases, change the format into float.
                        """
                        new_x.append(float(x_i))
                data_new.append(np.array(new_x))
        X = np.array(data_new)

        """
        Preprocess the test data in similar way.
        """
        test_new = []
        for z in test_data[1:]:
            new_z = []
            for i in range(len(z)):
                z_i = z[i].decode('utf8')
                if i == 0:
                    """
                    Feature pclass.
                    Handle the categorical feature by vectorizing it.
                    """
                    if z_i == '1.0':
                        new_z = new_z + [1, 0, 0]
                    elif z_i == '2.0':
                        new_z = new_z + [0, 1, 0]
                    elif z_i == '3.0':
                        new_z = new_z + [0, 0, 1]
                elif i == 2:
                    """
                    Feature age.
                    Handle the missing features by replacing the None
                    values with the medican among ones in the same class.
                    """
                    if not z_i:
                        new_z.append(med_age)
                    else:
                        new_z.append(float(z_i))
                elif i == 1:
                    """
                    Feature sex.
                    Map to 0 if male, 1 if Female.
                    """
                    if z_i == 'male':
                        new_z.append(0)
                    elif z_i == 'female':
                        new_z.append(1)

                elif i == 8:
                    """
                    Feature embarked.
                    Handle the categorical feature by vectorizing it.
                    """
                    if z_i == '':
                        z_i = maj_embarked
                    if z_i == 'C':
                        new_z = new_z + [1, 0, 0]
                    elif z_i == 'Q':
                        new_z = new_z + [0, 1, 0]
                    elif z_i == 'S':
                        new_z = new_z + [0, 0, 1]
                elif i == 6:
                    """
                    Feature fare.
                    Handle missing values by replacing it with median.
                    """
                    if z_i == '':
                        new_z.append(med_fare)
                    else:
                        new_z.append(float(z_i))
                elif i != 5 and i != 7:
                    """
                    Not use 'cabin' feature because there are too many missing values.
                    Not use 'ticket' features because the formats are all so different and
                    the ticket number seems to be independent with survival.
                    For other cases, change the format into float.
                    """
                    new_z.append(float(z_i))
            test_new.append(np.array(new_z))
        Z = np.array(test_new)
        """
        Convert y values to integers (0 or 1).
        """
        new_y = []
        for i in range(len(y)):
            new_y.append(int(y[i].decode('utf8')))
        y = new_y



    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam-dataset/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]
         
    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)
    

    spam_plot_depth_verror = False
    spam_kaggle = False
    spam_decision_tree = False
    spam_random_forrest = False
    titanic_decision_tree = False
    titaninc_random_forrest = False
    titanic_kaggle = False


    if spam_plot_depth_verror:
        print("Plotting Tree Depth Vs Validation Error for SPAM...")
        """
        Generate a random 80/20 training/validation split.
        """
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)
        
        i = math.ceil(len(y)*0.8)
        Xt = X[:i]
        yt = y[:i]
        Xv = X[i:]
        yv = y[i:]
        """
        Plot validation errors with different maximum depths (from 1 to 40).
        """
        plt.title("Depth VS Validation Error")
        plt.xlabel("depth")
        plt.ylabel("validiation error")
        depths = []
        v_errors = []
        for d in range(1, 61):
            classifier = DecisionTree(maxDepth=d)
            classifier.fit(Xt, yt)
            predictions = classifier.predict(Xv)
            v_error = classifier.errorRate(predictions, yv)
            depths.append(d)
            v_errors.append(v_error)
        plt.plot(depths, v_errors)
        plt.show()

    if spam_kaggle:
        print("Making Kaggle Predictions for Spam...")
        """
        Make predictions of the test data for kaggle submission.
        Set maxDepth=22 based on the validation test.
        """
        classifier = RandomForest(100, maxDepth=57)
        classifier.fit(X, y)
        predictions = np.array(classifier.predict(Z))
        results_to_csv(predictions)

    if spam_decision_tree:
        """
        Generate a random 80/20 training/validation split.
        """
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)
        
        i = math.ceil(len(y)*0.8)
        Xt = X[:i]
        yt = y[:i]
        Xv = X[i:]
        yv = y[i:]

        """
        Make predictions and check the validation error using Random Forrests.
        """
        classifier = DecisionTree(maxDepth=57)
        classifier.fit(Xt, yt)
        train_predictions = classifier.predict(Xt)
        val_predictions = classifier.predict(Xv)
        print("training error rate for spam DT: ", DecisionTree.errorRate(train_predictions, yt))
        print("validation error rate for spam DT: ", DecisionTree.errorRate(val_predictions, yv))


    if spam_random_forrest:
        """
        Generate a random 80/20 training/validation split.
        """
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)
        
        i = math.ceil(len(y)*0.8)
        Xt = X[:i]
        yt = y[:i]
        Xv = X[i:]
        yv = y[i:]

        """
        Make predictions and check the validation error using Random Forrests.
        """
        classifier = RandomForest(100, maxDepth=57)
        classifier.fit(Xt, yt)
        train_predictions = classifier.predict(Xt)
        val_predictions = classifier.predict(Xv)
        print("training error rate for spam RF: ", DecisionTree.errorRate(train_predictions, yt))
        print("validation error rate for spam RF: ", DecisionTree.errorRate(val_predictions, yv))

    if titanic_decision_tree:
        print("testing titanic decision tree...")
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)

        i = math.ceil(len(y)*0.8)
        Xt = X[:i]
        yt = y[:i]
        Xv = X[i:]
        yv = y[i:]

        classifier = DecisionTree(maxDepth=3)
        classifier.fit(Xt, yt)
        train_predictions = classifier.predict(Xt)
        val_predictions = classifier.predict(Xv)
        print("training error rate for titanic DT: ", DecisionTree.errorRate(train_predictions, yt))
        print("validation error rate for titanic DT: ", DecisionTree.errorRate(val_predictions, yv))

    if titaninc_random_forrest:
        print("testing titanic random forrest...")
        np.random.seed(42)
        np.random.shuffle(X)
        np.random.seed(42)
        np.random.shuffle(y)

        i = math.ceil(len(y)*0.8)
        Xt = X[:i]
        yt = y[:i]
        Xv = X[i:]
        yv = y[i:]

        classifier = RandomForest(100, maxDepth=60)
        classifier.fit(Xt, yt)
        train_predictions = classifier.predict(Xt)
        val_predictions = classifier.predict(Xv)
        print("training error rate for titanic RF: ", DecisionTree.errorRate(train_predictions, yt))
        print("validation error rate for titanic RF: ", DecisionTree.errorRate(val_predictions, yv))

    if titanic_kaggle:
        classifier = RandomForest(500, maxDepth=20)
        classifier.fit(X, y)
        predictions = np.array(classifier.predict(Z))
        results_to_csv(predictions)
