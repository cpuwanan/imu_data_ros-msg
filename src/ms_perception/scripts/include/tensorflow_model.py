#!/home/msasrock/.virtualenvs/ros-melodic-venv/bin/python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random
import io

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def helloworld():
  print("Welcome to TF {}".format(tf.__version__))

class MyIMUNet:
  def __init__(self, results_dir, prefix):
    self.__dir = results_dir
    self.checkpoint_path = "{}/{}_weights/cp.ckpt".format(self.__dir, prefix)
    self.model_filename = "{}/{}_model".format(self.__dir, prefix)
    self.figure_filename = "{}/{}_evaluate.png".format(self.__dir, prefix)
    self.report_filename = "{}/{}_report.txt".format(self.__dir, prefix)
    print("Tensorflow files at \n -- {}\n -- {}\n -- {}\n -- {}".format(
      self.checkpoint_path,
      self.model_filename,
      self.figure_filename,
      self.report_filename
    ))

  def onHotEncoding(self, label):
    return keras.utils.to_categorical(label)

  def loadModel(self, singleInputShape, classNum, loadWeightsOnly = False, report_filename = None):
    if loadWeightsOnly:
      self.__build(singleInputShape, classNum)
      self.__compile()
      self.model.load_weights(self.checkpoint_path)
    else:
      self.model = keras.models.load_model(self.model_filename)
      self.model.summary()
      self.model.load_weights(self.checkpoint_path)
    
    summary = self.__getModelSummary(self.model)
    self.__saveReportFile(summary, [], report_filename)

  def train(self, X_train, y_train, X_test, y_test, epochNum):
    print("About to be training")
    print(" -- Total number of classes: {}".format(len(y_train[0])))
    print(" -- Single input shape: {}".format(X_train[0].shape))
    print(" -- Single label shape: {}".format(y_train[0].shape))
    print(" -- Total data. Train: {}, Test: {}".format(len(X_train), len(X_test)))
        
    self.__build(X_train[0].shape, len(y_train[0]))
    self.__compile()
    
    # create a callback that saves the model's weight
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=self.checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    # Traing the model with the new callback
    self.model.fit(
      X_train, y_train, 
      validation_data=(X_test, y_test), 
      callbacks=[cp_callback],
      epochs=epochNum
    )

    self.model.save(self.model_filename)

    summary = self.__getModelSummary(self.model)
    
    test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)
    extra = np.array(["test_accuracy: {:.3f}".format(test_acc)])
    extra = np.append(extra, ["test_loss: {:.3f}".format(test_loss)])
    extra = np.append(extra, ["epochs_num: {}".format(epochNum)])
    extra = np.append(extra, ["datasets_shapes: {}, {}".format(X_train.shape, y_train.shape)])
    extra = np.append(extra, ["single_data_shapes: {}, {}".format(X_train[0].shape, y_train[0].shape)])
    extra = np.append(extra, ["single_one_hot_encoding: {}".format(y_train[0])])
    extra = np.append(extra, ["train_test_split ({}, {}): {:.3f}".format(len(X_train), len(X_test), len(X_test) / len(X_train))])
    self.__saveReportFile(summary, extra, None)
    
  def predict(self, X):
    if self.model is None: 
      return None
    else:
      Z = self.model.predict(X)
      return Z

  def evaluate(self, X_test, y_test, events):
    test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)
    print(" -- Evaluate. Loss: {}, Accuracy: {}".format(test_loss, test_acc))
    
    start_index = random.randint(0, len(X_test))
    
    predictions_all_data = self.model.predict(X_test[start_index: start_index + 2])
    fix, axes = plt.subplots(2, 2)
    counter = 0
    for predictions_array in predictions_all_data:
      # Show image
      predicted_label = np.argmax(predictions_array)
      true_label = np.argmax(y_test[start_index + counter])

      print("{} >> {} >> predicted: {} ({})".format(start_index, predictions_array, predicted_label, events[predicted_label]))  
      image = X_test[start_index + counter].reshape(X_test[start_index + counter].shape[0], X_test[start_index + counter].shape[1])
      axes[counter, 0].imshow(image, cmap='gray')
      axes[counter, 0].grid(True)
      axes[counter, 0].axis('off')
      axes[counter, 0].set_title("Id: {}, predict: {} ({})".format(start_index + counter, predicted_label, events[predicted_label]))
      
      # Show predictions
      axes[counter, 1].grid(True)
      N = len(y_test[0])
      axes[counter, 1].set_ylabel("Probability")
      # axes[counter, 1].set_xticks(np.arange(N))
      axes[counter, 1].set_xticks(np.arange(N))
      axes[counter, 1].set_xticklabels(events)
      axes[counter, 1].set_yticks(np.arange(0.0, 1.0, 0.1))
      myplot = axes[counter, 1].bar(range(N), predictions_array, color="#777777")
      axes[counter, 1].set_ylim([0, 1])
      myplot[predicted_label].set_color('red')
      axes[counter, 1].set_title("True: {} ({}), prob: {:.3f}".format(true_label, events[true_label], predictions_array[true_label]))
      counter += 1
    plt.tight_layout()
    plt.savefig(self.figure_filename, format="png")
    plt.show()

  # Private functions

  def __build(self, shape, classNum):
    self.model = Sequential()
    self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=shape))
    self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
    self.model.add(Flatten())
    self.model.add(Dense(classNum, activation='softmax'))
  
  def __compile(self):
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    self.model.summary()
    return self.model
  
  def __getModelSummary(self, model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

  def __saveReportFile(self, data, extra = [], filename = None):
    if filename is not None:
      self.report_filename = filename 
    file = open(self.report_filename, "w")
    file.write(data)
    if len(extra) > 0:
      file.write("\nExtra Information:\n")
      for k in range(len(extra)):
          file.write(" -- {}\n".format(extra[k]))
    file.close()
    print("Save info to '{}' successfully".format(self.report_filename))
