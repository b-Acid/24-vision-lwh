#!/usr/bin/python3
import cv2
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

 
if __name__ == '__main__':
    rospy.init_node("camera_node")
    image_pub=rospy.Publisher('/image_view/image_raw', Image, queue_size = 1) #定义话题
    rate = rospy.Rate(5)
    capture = cv2.VideoCapture(0)
    ros_frame = Image()
    header = Header(stamp = rospy.Time.now())
    rospy.loginfo("Caputuring...." ) 
    while not rospy.is_shutdown():
        ret, frame = capture.read()
        if ret: # 如果有画面再执行
            header.stamp = rospy.Time.now()
            header.frame_id = "Camera"
            ros_frame.header=header
            ros_frame.width = 640
            ros_frame.height = 480
            ros_frame.encoding = "bgr8"
            ros_frame.step = 1920
            ros_frame.data = np.array(frame).tostring() #图片格式转换
            image_pub.publish(ros_frame) #发布消息  
        rate.sleep()
    rospy.loginfo("Camera shutdown!" ) 
