#!/home/msasrock/.virtualenvs/ros-melodic-venv/bin/python
import rospy
import rospkg
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from ms_msgs.msg import IMUData
from sensor_msgs.msg import JointState

sys.path.append(os.path.join(os.path.dirname(__file__), 'include'))
import include.tensorflow_model as mytf_model
import include.config_loader as config_loader
import include.viz_utils as viz_utils

LIGHT_GREEN='\033[1;32m'
NC='\033[0m' # No Color

class MyRobot:
	def __init__(self):
		config_filename = rospy.get_param("~config_filename", None)
		if (config_filename is None):
			rospy.logerr("Invalid config filename: {}".format(config_filename))
			rospy.is_shutdown()
			return
		
		rospack = rospkg.RosPack()
		self.__root_dir = rospack.get_path("ms_perception")

		self.__prob_threshold = rospy.get_param("~prob_threshold", 0.8)
		self.__show_scorebar = rospy.get_param("~show_scorebar", True)
		
		config = config_loader.YAMLReader(config_filename)
		self.__target_indices, self.__target_names = config.getFullIndexConfig("selective_data")
		self.__annotations = config.getEventAnnotation("event_annotations")
		rospy.loginfo("Annotations: {}".format(self.__annotations))
		
		data_dir = self.__root_dir + "/config"
		saved_dir = config.getSublevelData("tensorflow", "saved_dir")
		self.__data_shape = config.getSublevelData("tensorflow", "data_shape")
		self.__data_shape = (self.__data_shape[0], self.__data_shape[1], 1)
		class_num = len(self.__annotations)

		rospy.loginfo("Input data shape: {}, Total class num: {}".format(self.__data_shape, class_num))
		rospy.loginfo("Model dir: {}".format(data_dir + saved_dir))

		report_filename = self.__root_dir + "/tf_imu_model_summary.txt"
		self.__net = mytf_model.MyIMUNet(data_dir + saved_dir, "tf_imu")
		self.__net.loadModel(self.__data_shape, class_num, loadWeightsOnly=False, report_filename=report_filename)

		self.__colors = [viz_utils.LIGHT_CYAN, viz_utils.LIGHT_PURPLE, viz_utils.LIGHT_GREEN, viz_utils.LIGHT_YELLOW, viz_utils.LIGHT_RED, viz_utils.LIGHT_BLUE]

		self.__imu_data_sub = rospy.Subscriber("imu_data", IMUData, self.imuDataCallback)		
		self.__detections_pub = rospy.Publisher('detections', JointState, queue_size=1)

		self.__predictions = None
		self.__input_data = None

	def run(self):
		if self.__show_scorebar is True:
			rospy.logwarn("Display the score bar in realtime. Beware that you may have the lacking issues while plotting.")
			self.__showScorebar()
		else:
			rospy.loginfo("Looks good !")
			rospy.spin()

	def imuDataCallback(self, data):
		preprocessed = self.__prepare(data)
		if preprocessed is not None:
			preprocessed = np.array([preprocessed])
			Z = self.__net.predict(preprocessed)
			predicted_label = np.argmax(Z[0])
			rospy.loginfo("Predicted class: {}{}{} ({:.2f}%)".format(self.__colors[predicted_label], self.__annotations[predicted_label], viz_utils.NC, Z[0][predicted_label] * 100))
			
			res_names = np.array([self.__annotations])
			res_values = np.array([Z[0]])
			
			results = JointState()
			results.header.frame_id = rospy.get_name()
			results.header.stamp = data.header.stamp
			results.name = np.array([])
			results.position = np.array([])
			results.name = np.append(results.name, self.__annotations, axis=0)
			results.position = np.append(results.position, Z[0], axis=0)
			self.__detections_pub.publish(results)

			self.__predictions = Z[0]
			self.__input_data = preprocessed 
		else:
			rospy.logwarn("[seq {}] Awaiting for valid data".format(data.header.seq))

	def __prepare(self, data):
		values = np.array([data.ax, data.ay, data.az, data.gx, data.gy, data.gz])
		values = values.reshape(values.shape[0], values.shape[1], 1)
		if (values.shape == self.__data_shape):
			return values
		return None

	def __showScorebar(self):
		rate = rospy.Rate(10)
		fig, axes = plt.subplots(1, 2)
		is_plot_available = False
		
		while not rospy.is_shutdown():
			if (self.__predictions is not None) and (self.__input_data is not None):
				input_data = self.__input_data
				input_data = input_data.reshape(input_data.shape[1], input_data.shape[2])
				predictions = self.__predictions
				predicted_label = np.argmax(predictions)

				# Show image
				index = 0
				im = axes[index].imshow(input_data, cmap='gray')
				axes[index].grid(True)
				axes[index].axis('off')
				axes[index].set_title("Predicted: {}".format(self.__annotations[predicted_label]))
				
				# Show predictions
				index = 1
				axes[index].cla()
				axes[index].grid(True)
				N = len(self.__annotations)
				axes[index].set_ylabel("Probability")
				axes[index].set_xticks(np.arange(N))
				axes[index].set_xticklabels(self.__annotations)
				axes[index].set_yticks(np.arange(0.0, 1.0, 0.1))
				myplot = axes[index].bar(range(N), predictions, color="#777777")
				axes[index].set_ylim([0, 1])
				myplot[predicted_label].set_color('red')
				axes[index].set_title("score: {:.3f}".format(predictions[predicted_label]))

				# Show all
				plt.tight_layout()
				plt.pause(0.01)
				is_plot_available = True

			rate.sleep()
		
		if is_plot_available:
			filename = self.__root_dir + "/tf_imu_predict.png"
			plt.savefig(filename, format="png")
			print("Saved chart at {}".format(filename))

def main():
  rospy.init_node('hello_world', anonymous=True)
  robot = MyRobot()
  robot.run()

if __name__ == "__main__":
  try:
    main()
  except rospy.ROSInterruptException:
    pass
