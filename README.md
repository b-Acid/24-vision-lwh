# 24-vision-lwh
华南虎机器人实验室视觉组实习任务

###  1.[装甲板识别任务](https://github.com/b-Acid/24-vision-lwh/tree/main/%E8%A3%85%E7%94%B2%E6%9D%BF%E8%AF%86%E5%88%AB)
+ 基于OpenCV 4.7.0，通过轮廓检测，pnp等方法，实现了对装甲板的检测和姿态解算。

###  2.[装甲板分类任务](https://github.com/b-Acid/24-vision-lwh/tree/main/%E8%A3%85%E7%94%B2%E6%9D%BF%E5%88%86%E7%B1%BB)
+ 基于pytorch机器学习框架搭建了resnet18进行装甲板分类,使用软件的版本是torch2.0.1+cuda11.7+torchvision0.15.2。

### 3.[SSH任务](https://github.com/b-Acid/24-vision-lwh/tree/main/SSH%E4%BB%BB%E5%8A%A1)

+ 使用SSH进行两机间互传文件，编写configure文件使用别名实现快速登录。

### 4.[Cmake I任务（也有Cmake II）](https://github.com/b-Acid/24-vision-lwh/tree/main/Cmake%E4%BB%BB%E5%8A%A1)

+ 首先在子文件夹编写CMakeLists.txt，将子目录里的程序编译成静态库，之后在主目录的CMakeLists.txt指定了所有头文件的路径和要链接的库，完成主函数的编译。

### 5.[onnx部署模型任务](https://github.com/b-Acid/24-vision-lwh/tree/main/onnx%E9%83%A8%E7%BD%B2)
+ 使用onnx-python将模型转化为onnx模式，并使用onnx-simplifier进行简化。用onnxruntime在c++中部署onnx模型。这里使用的是基于wideface数据集训练的只检测人脸的yolov5模型。使用的onnxruntime版本是linux-gpu1.16.0。使用的pytorch和cuda版本同任务2。RTX3060上部署yolov5模型推理320*320的图片耗时1ms左右。
  
### 6.[tensorRT部署模型任务](https://github.com/b-Acid/24-vision-lwh/tree/main/tensorRT%E9%83%A8%E7%BD%B2)
+ 使用tensorRT提供的API将onnx模型转化为engine模式，并使用tensorRT的API加载engine，创建推理stream进行推理。使用的tensorRT版本是8.6.1。RTX3060上部署yolov5模型推理320*320的图片耗时1ms左右。
  
### 7.[点云投影任务](https://github.com/b-Acid/24-vision-lwh/tree/main/%E7%82%B9%E4%BA%91%E6%8A%95%E5%BD%B1%E4%BB%BB%E5%8A%A1)
+ 使用相机内外参矩阵将世界坐标中的点（x，y，z）投影到像素坐标上（u，v），绘制这些点构成点云，描绘背景轮廓。结果保存在outputs里。

### 8.[ROS通信](https://github.com/b-Acid/24-vision-lwh/tree/main/ros%E9%80%9A%E4%BF%A1)
+ 使用ros新建了一个package为camera_driver,实现了创建一个名为camera_node的节点，该节点通过话题image发布。其他节点可以订阅image话题来获取摄像头画面。
