#!/home/msasrock/.virtualenvs/ros-melodic-venv/bin/python
import numpy as np
import yaml

class YAMLReader:
  def __init__(self, filename):
    with open(filename) as file:
      self.__node = yaml.load(file, Loader=yaml.FullLoader)

  def getIndexConfig(self, key):
    selective_indices = np.array([])
    if self.__node[key]:
      for data in self.__node[key]:
        selective_indices = np.append(selective_indices, [data["index"]])
    return selective_indices
    
  def getFullIndexConfig(self, key):
    indices = np.array([])
    names = np.array([])
    if self.__node[key]:
      for data in self.__node[key]:
        names = np.append(names, [data["name"]])
        if len(indices) == 0:
          indices = np.array([data["index"]])
        else:
          indices = np.append(indices, [data["index"]], axis=0)
    return indices, names

  def getStringData(self, key):
    if self.__node[key]:
      return self.__node[key]
    return None

  def getEventAnnotation(self, key):
    if self.__node[key]:
      return self.__node[key]
    return None

  def getDatanames(self, key):
    datanames = np.array([])
    if self.__node[key]:
      for data in self.__node[key]:
        indices = data["index"]
        for k in range(len(indices)):
          if indices[k] == 1:
            name = "{}-{}".format(data["name"], k)
            datanames = np.append(datanames, [name])
    return datanames
  
  def getSublevelData(self, key1, key2):
    if (self.__node[key1]):
      if self.__node[key1][key2]:
        return self.__node[key1][key2]
    print("Invalid config keys for '{}', '{}'".format(key1, key2))
    return None
