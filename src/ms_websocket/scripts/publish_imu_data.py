#!/usr/bin/env python
import rospy
import numpy as np
import random
from ms_msgs.msg import IMUData, ConfusionMatrixArray, ConfusionMatrix

def checkArraySize(array):
    if len(array) >= 11:
        array = array[1 : 11]
    return array

def getRandomMatrix(title, row_title, col_title, rows, cols, time):
    res = ConfusionMatrix()
    res.header.stamp = time
    res.header.frame_id = rospy.get_name()
    res.title = title
    res.row_title = row_title
    res.col_title = col_title
    res.rows = rows
    res.cols = cols
    res.data = np.array([])
    for i in range(rows * cols):
        res.data = np.append(res.data, [random.random()])
    return res

def talker():
    imu_pub = rospy.Publisher('imu_data', IMUData, queue_size=1)
    matrices_pub = rospy.Publisher('confusion_matrix_array', ConfusionMatrixArray, queue_size=1)

    rospy.init_node('talker', anonymous=True)

    ax_array = np.array([])
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():

        now = rospy.Time.now()

        # Sampling IMU data
        ax_t = random.random()
        ax_array = np.append(ax_array, [ax_t])    
        ax_array = checkArraySize(ax_array)
        imu_data_msg = IMUData()
        imu_data_msg.header.frame_id = rospy.get_name()
        imu_data_msg.header.stamp = now
        imu_data_msg.time = []
        imu_data_msg.ax = ax_array
        imu_pub.publish(imu_data_msg)
        
        # Sampling matrix
        matrices_msg = ConfusionMatrixArray()
        matrices_msg.header.frame_id = rospy.get_name()
        matrices_msg.header.stamp = now
        matrices_msg.matrices = np.array([])
        mat1 = getRandomMatrix(
                "aX versus aY",
                "aY", "aX", 10, 10, now
                )
        mat2 = getRandomMatrix(
                "aY versus aZ",
                "aZ", "aY", 10, 10, now
            )
        matrices_msg.matrices = np.append(matrices_msg.matrices, [mat1], axis=0)
        matrices_msg.matrices = np.append(matrices_msg.matrices, [mat2], axis=0)

        matrices_pub.publish(matrices_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
