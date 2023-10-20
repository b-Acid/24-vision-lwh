# ROS通信任务
+ 本任务完成了创建一个ros节点，在该节点上打开摄像头并发布摄像头画面。

### 具体实现
1. 创建一个ros工作区，这里命名为catkin_ws，并在这个目录下创建源代码文件夹src
```
mkdir -p catkin_ws/src
```
2. 进入工作区的 src 目录， 使用 catkin_init_workspace 命令初始化工作区。
```
cd catkin_ws/src
catkin_init_workspace
```

3.返回工作区主目录,使用 catkin_make 命令构建工作区。这将编译ROS软件包并生成构建文件。
```
cd ..
catkin_make
```

4.再切到src目录。创建一个包camera_driver
```
catkin_create_pkg camera_driver
```

5.创建一个camera.launch里面设置了要启动的节点和一些参数
```
<launch>
     <node pkg="camera_driver" type="camera_node.py" name="camera_node" output="screen">
       <!-- 相机参数 -->
       <param name="camera_topic" type="string" value="image "/>
       <param name="camera_resolution" type="string" value="640x480" />
       <param name="camera_fps" type="int" value="30" />
     </node>
</launch>
```

6.接着上一步，上一步指定了执行文件是camera_node.py，创建它，写入要运行的程序。程序的主要结构如下，它实现了初始化节点并打开一个话题为image，这个话题发布了图像信息。
```python
import cv2
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

 
if __name__ == '__main__':
    rospy.init_node("camera_node")#初始化节点
    image_pub=rospy.Publisher('image', Image, queue_size = 1) #定义话题
    rate = rospy.Rate(5)#设置发布频率
    capture = cv2.VideoCapture(0)#开摄像头
    ros_frame = Image()#创建一个Image对象也就是等下要发布的
    header = Header(stamp = rospy.Time.now())
    rospy.loginfo("Caputuring...." ) 
    while not rospy.is_shutdown():
        ret, frame = capture.read()
        if ret: # 如果有画面再执行
            #处理报头
            #处理ros_frame
            #发布消息
            image_pub.publish(ros_frame)  
        rate.sleep()
    rospy.loginfo("Camera shutdown!" ) 
```
7.回到工作环境主目录，执行命令catkin_make完成编译。可能会报错是因为ros默认python2但是电脑只有python3,指明python路径即可。
```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```
8.执行下面的命令来运行程序。
```
source devel/setup.bash
roslaunch camera_driver camera.launch 
```
+ 这时摄像头灯光打开，摄像头开始运行，图像信息通过camera_node发布出来。
+ 输入 rostopic list看下哪些topic正在发布，可以看到刚刚打开的话题image正在运行。
  ![](https://github.com/b-Acid/Images/blob/main/ros%E9%80%9A%E4%BF%A1/111.png?raw=true)
+ 现在在其他ros程序上就可以订阅这个话题，接收到摄像头信息了。
