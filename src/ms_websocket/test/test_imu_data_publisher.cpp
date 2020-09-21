#include <ros/ros.h>
#include <ms_msgs/IMUData.h>
#include <ms_msgs/ConfusionMatrix.h>
#include <ms_msgs/ConfusionMatrixArray.h>

#include <iostream>
#include <vector>

template <class T>
void checkArraySize(std::vector<T> &vec, int N) {
	if (vec.size() > N) {
		int len = vec.size() - N;
		vec.erase(vec.begin(), vec.begin() + len);
	}
}

ms_msgs::ConfusionMatrix getRandomMatrix(std::string title, std::string row_title, std::string col_title, int rows, int cols, ros::Time time) {
	ms_msgs::ConfusionMatrix msg;
	msg.header.frame_id = ros::this_node::getName();
	msg.header.stamp = time;
	msg.title = title;
	msg.row_title = row_title;
	msg.col_title = col_title;
	msg.rows = rows;
	msg.cols = cols;
	msg.data.resize(rows * cols);
	int index = 0;
	for (int j=0; j<rows; j++) {
		for (int i=0; i<cols; i++, index++) {
			msg.data[index] = double(rand() % 100 / 100.f);
		}
	}
	return msg;
}

int main(int argc, char ** argv) {
	ros::init(argc, argv, "test");
	ros::NodeHandle nh, private_nh("~");
	
	std::string imu_data_topic, conf_mat_array_topic;
	private_nh.param<std::string>("imu_data_topic", imu_data_topic, "imu_data");
	private_nh.param<std::string>("confusion_matrix_array_topic", conf_mat_array_topic, "confusion_matrix_array");
		
	ros::Publisher imu_data_pub = nh.advertise<ms_msgs::IMUData>(imu_data_topic, 1);
	ros::Publisher conf_mat_array_pub = nh.advertise<ms_msgs::ConfusionMatrixArray>(conf_mat_array_topic, 1);
	
	int N = 10;
	std::vector<float> time_array;
	std::vector<float> ax_array, ay_array;
	
	float ax_t = rand() % 100;
	float ay_t = rand() % 100;
	
	ros::Time t_begin = ros::Time::now();
	ros::Rate rate(3.0);
	while (ros::ok()) {
		
		float time_t = (ros::Time::now() - t_begin).toSec();
		ax_t += (rand() % 10 - 5.0);
		ay_t += (rand() % 10 - 5.0);
		
		time_array.push_back(time_t);
		ax_array.push_back(ax_t);
		ay_array.push_back(ay_t);

		checkArraySize(time_array, N);
		checkArraySize(ax_array, N);
		checkArraySize(ay_array, N);
		
		ros::Time now = ros::Time::now();
		
		ms_msgs::IMUData imu_data;
		imu_data.header.frame_id = imu_data_topic;
		imu_data.header.stamp = now;
		imu_data.time = time_array;
		imu_data.ax = ax_array;
		imu_data.ay = ay_array;
		imu_data_pub.publish(imu_data);
		
		ms_msgs::ConfusionMatrixArray array_data;
		array_data.header.frame_id = conf_mat_array_topic;
		array_data.header.stamp = now;
		array_data.matrices.push_back(getRandomMatrix(
			"aX versus aY", "aX", "aY", 10, 10, now
		));
		array_data.matrices.push_back(getRandomMatrix(
			"aY versus aZ", "aY", "aZ", 10, 10, now
		));
		array_data.matrices.push_back(getRandomMatrix(
			"aZ versus ax", "aZ", "aX", 10, 10, now
		));		
		conf_mat_array_pub.publish(array_data);
		
		rate.sleep();
		ros::spinOnce();
	}
	
	return 0;
}
