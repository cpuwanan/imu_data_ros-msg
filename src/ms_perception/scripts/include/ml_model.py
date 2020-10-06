#!/home/msasrock/.virtualenvs/ros-melodic-venv/bin/python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pickle
from datetime import date

def helloworld():
	print("Welcome to ML-Model")

class MLModel:
  def __init__(self):
    self.__model = None
    self.__name = "undefined"
    self.__score = 0.0
    return None

  def prepare(self, data, target, test_size = 0.3, random_state = 4):
    self.__X_train, self.__X_test, self.__Y_train, self.__Y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    return self.__X_train, self.__X_test, self.__Y_train, self.__Y_test

  def train(self, model_name):
    self.__name = model_name
    if model_name == "gaussian":
      self.__model, self.__score = self.__gaussianProcessClassifier()
    elif model_name == "svm":
      self.__model, self.__score = self.__svm()
    elif model_name == "neuralnet":
      self.__model, self.__score = self.__neuralnet()
    elif model_name == "naivebayes":
      self.__model, self.__score = self.__naivebayes()

    if self.__score > 0.0:
      print("Train by '{}' method. Accuracy: {:.3f}".format(self.__name, self.__score))
      return True

    return False

  def predict(self, X):
    if self.__model is None:
      return []
    Z = self.__model.predict_proba(X)
    return Z

  def load(self, filename):
    with open(filename, 'rb') as file:
      self.__model = pickle.load(file)
      return True
    return False

  def save(self, root_dir, prefix_name):

    if self.__model is None:
      print("Error. No model available")
      return False

    pkl_filename = "{}.pkl".format(prefix_name)
    # Save to file
    with open(root_dir + "/" + pkl_filename, 'wb') as file:
      pickle.dump(self.__model, file)

    today = date.today()
    # dd/mm/YY
    d1 = today.strftime("%d/%m/%Y")

    info_filename = "{}.txt".format(prefix_name)
    file = open(root_dir + "/" + info_filename, "w")
    file.write("model_name: {}\n".format(self.__name))
    file.write("filename: {}\n".format(pkl_filename))
    file.write("accuracy: {}\n".format(self.__score))
    file.write("train_size: {}\n".format(len(self.__X_train)))
    file.write("test_size: {}\n".format(len(self.__X_test)))
    file.write("date: {}\n".format(d1))
    file.close()
    
    print("Saved successfully: \n -- Model: {}\n -- Info: {}".format(pkl_filename, info_filename))
    return True

  ## Private functions

  def __gaussianProcessClassifier(self):
    model = GaussianProcessClassifier(1.0 * RBF(1.0))
    model.fit(self.__X_train, self.__Y_train)
    score = model.score(self.__X_test, self.__Y_test)
    return model, score

  def __svm(self):
    model = SVC(gamma=2, C=1)
    model.fit(self.__X_train, self.__Y_train)
    score = model.score(self.__X_test, self.__Y_test)
    return model, score

  def __neuralnet(self):
    model = MLPClassifier(alpha=1, max_iter=1000)
    model.fit(self.__X_train, self.__Y_train)
    score = model.score(self.__X_test, self.__Y_test)
    return model, score

  def __naivebayes(self):
    model = GaussianNB()
    model.fit(self.__X_train, self.__Y_train)
    score = model.score(self.__X_test, self.__Y_test)
    return model, score
