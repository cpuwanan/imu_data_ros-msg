#!/home/msasrock/.virtualenvs/ros-melodic-venv/bin/python
import rospy
import sys, os
import numpy as np

from ms_msgs.msg import IMUData

sys.path.append(os.path.join(os.path.dirname(__file__), 'include'))
import include.ml_model as ml_model
import include.config_loader as config_loader
import include.viz_utils as viz_utils

LIGHT_GREEN='\033[1;32m'
NC='\033[0m' # No Color

class MyRobot:
	def __init__(self):
		model_filename = rospy.get_param("~model_filename", None)
		if (model_filename is None):
			rospy.logerr("Invalid model filename: {}".format(model_filename))
			rospy.is_shutdown()
			return
		
		config_filename = rospy.get_param("~config_filename", None)
		if (model_filename is None):
			rospy.logerr("Invalid config filename: {}".format(config_filename))
			rospy.is_shutdown()
			return
		
		self.__prob_threshold = rospy.get_param("~prob_threshold", 0.8)
		
		config = config_loader.YAMLReader(config_filename)
		self.__target_indices, self.__target_names = config.getFullIndexConfig("selective_data")
		self.__annotations = config.getEventAnnotation("event_annotations")
		rospy.loginfo("Annotations: {}".format(self.__annotations))
		
		self.__model = ml_model.MLModel()
		res = self.__model.load(model_filename)
		rospy.loginfo("Loaded model successfully ? {}".format(res))
		self.__imu_data_sub = rospy.Subscriber("imu_data", IMUData, self.imuDataCallback)
		
		self.__colors = [viz_utils.LIGHT_GREEN, viz_utils.LIGHT_YELLOW, viz_utils.LIGHT_BLUE, viz_utils.LIGHT_RED, viz_utils.LIGHT_PURPLE]

		
	def imuDataCallback(self, data):		
		preprocessed = self.__prepare(data)
		if preprocessed is not None:
			Z = self.__model.predict([preprocessed])
			if len(Z) > 0:
				id = np.argmax(Z[0])
				if Z[0][id] > self.__prob_threshold:
					rospy.loginfo("{} >> {}{}{}".format(Z[0], self.__colors[id], self.__annotations[id], NC))
				else:
					rospy.loginfo("{}".format(Z[0]))
		else:
			rospy.logwarn("[seq {}] Awaiting for valid data".format(data.header.seq))
		
	def __prepare(self, data):
		values = np.array([])
		for ii in range(len(self.__target_names)):
			indices = self.__target_indices[ii]
			measured = np.array([])
			if ii == 0:
				measured = np.array(data.ax)
			elif ii == 1:
				measured = np.array(data.ay)
			elif ii == 2:
				measured = np.array(data.az)
			elif ii == 3:
				measured = np.array(data.gx)
			elif ii == 4:
				measured = np.array(data.gy)
			elif ii == 5:
				measured = np.array(data.gz)
			
			if len(measured) == 0:
				return None
			
			if len(measured) != len(indices):
				return None
			
			if len(measured) != len(indices):
				rospy.logerr("Wrong data size. Measured: {}, {}, Indices: {}".format(len(measured), measured, len(indices)))
				return None
			
			for flag, val in zip(indices, measured):
				if flag == 1:
					values = np.append(values, [val])

		return values

def main():
  rospy.init_node('hello_world', anonymous=True)
  robot = MyRobot()
  rospy.spin()

if __name__ == "__main__":
  try:
    main()
  except rospy.ROSInterruptException:
    pass
